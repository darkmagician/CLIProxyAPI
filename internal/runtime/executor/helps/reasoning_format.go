package helps

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// ReasoningFormatHeader selects the reasoning dialect used for historical
// thinking on the Claude<->OpenAI chat translation paths.
const ReasoningFormatHeader = "X-Reasoning-Format"

const (
	reasoningFormatOpenRouter = "openrouter"
	reasoningFormatDeepSeek   = "deepseek"
)

// normalizedReasoningFormat maps the client's X-Reasoning-Format header to a
// reasoning dialect. "openrouter" selects the OpenRouter-style reasoning
// field; anything else (deepseek, missing, unknown) falls back to the
// DeepSeek-style reasoning_content as the default dialect.
func normalizedReasoningFormat(headers http.Header) string {
	value := strings.ToLower(strings.TrimSpace(HeaderValueCaseInsensitive(headers, ReasoningFormatHeader)))
	if value == reasoningFormatOpenRouter {
		return reasoningFormatOpenRouter
	}
	return reasoningFormatDeepSeek
}

// ApplyReasoningFormatFromHeaders renames assistant reasoning_content into the
// OpenRouter-style reasoning field for translated Claude->OpenAI chat payloads
// when the client requests the openrouter dialect. Other dialects keep
// reasoning_content untouched.
func ApplyReasoningFormatFromHeaders(headers http.Header, payload []byte) []byte {
	if normalizedReasoningFormat(headers) != reasoningFormatOpenRouter {
		return payload
	}
	messages := gjson.GetBytes(payload, "messages")
	if !messages.IsArray() {
		return payload
	}
	out := payload
	messageIndex := 0
	messages.ForEach(func(_, message gjson.Result) bool {
		if message.Get("role").String() == "assistant" {
			if reasoning := message.Get("reasoning_content"); reasoning.Exists() {
				if updated, errSet := sjson.SetBytes(out, fmt.Sprintf("messages.%d.reasoning", messageIndex), reasoning.String()); errSet == nil {
					out = updated
				}
				if updated, errDel := sjson.DeleteBytes(out, fmt.Sprintf("messages.%d.reasoning_content", messageIndex)); errDel == nil {
					out = updated
				}
			}
		}
		messageIndex++
		return true
	})
	return out
}

// ApplyReasoningFormatToResponseFromHeaders strips the non-preferred reasoning
// field from an upstream OpenAI chat response (streaming delta or non-streaming
// message) before response translation. The openrouter dialect keeps only the
// reasoning field; every other dialect keeps only reasoning_content. Deleting
// the non-preferred field lets the translator's fallback resolve exactly the
// requested field.
func ApplyReasoningFormatToResponseFromHeaders(headers http.Header, payload []byte) []byte {
	if len(payload) == 0 {
		return payload
	}
	choices := gjson.GetBytes(payload, "choices")
	if !choices.IsArray() {
		return payload
	}
	field := "reasoning"
	if normalizedReasoningFormat(headers) == reasoningFormatOpenRouter {
		field = "reasoning_content"
	}
	out := payload
	choices.ForEach(func(key, _ gjson.Result) bool {
		for _, container := range []string{"delta", "message"} {
			path := fmt.Sprintf("choices.%d.%s.%s", key.Int(), container, field)
			if gjson.GetBytes(out, path).Exists() {
				if updated, errDel := sjson.DeleteBytes(out, path); errDel == nil {
					out = updated
				}
			}
		}
		return true
	})
	return out
}
