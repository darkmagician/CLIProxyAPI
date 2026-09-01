package helps

import (
	"net/http"
	"testing"

	"github.com/tidwall/gjson"
)

func TestApplyReasoningFormatFromHeaders(t *testing.T) {
	input := `{"model":"gpt-5","messages":[
		{"role":"user","content":"hi"},
		{"role":"assistant","content":"","reasoning_content":"step one"},
		{"role":"assistant","content":"answer"},
		{"role":"assistant","content":"","reasoning_content":"step two","tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]}
	]}`

	tests := []struct {
		name        string
		headers     http.Header
		wantField   string // "" asserts the field must not exist
		wantMissing string // field that must not exist on messages.1
	}{
		{name: "no header keeps reasoning_content", headers: nil, wantField: "reasoning_content", wantMissing: "reasoning"},
		{name: "deepseek keeps reasoning_content", headers: http.Header{"X-Reasoning-Format": {"deepseek"}}, wantField: "reasoning_content", wantMissing: "reasoning"},
		{name: "unknown value keeps reasoning_content", headers: http.Header{"X-Reasoning-Format": {"perplexity"}}, wantField: "reasoning_content", wantMissing: "reasoning"},
		{name: "openrouter renames to reasoning", headers: http.Header{"X-Reasoning-Format": {"openrouter"}}, wantField: "reasoning", wantMissing: "reasoning_content"},
		{name: "case and space tolerant", headers: http.Header{"X-Reasoning-Format": {"  OpenRouter  "}}, wantField: "reasoning", wantMissing: "reasoning_content"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ApplyReasoningFormatFromHeaders(tt.headers, []byte(input))
			// messages.1 carries reasoning_content in the input.
			if got := gjson.GetBytes(result, "messages.1."+tt.wantField).String(); got != "step one" {
				t.Fatalf("messages.1.%s = %q, want step one. Output: %s", tt.wantField, got, string(result))
			}
			if gjson.GetBytes(result, "messages.1."+tt.wantMissing).Exists() {
				t.Fatalf("messages.1.%s must not exist. Output: %s", tt.wantMissing, string(result))
			}
			// messages.3 (assistant with reasoning + tool_calls) follows the same dialect.
			if got := gjson.GetBytes(result, "messages.3."+tt.wantField).String(); got != "step two" {
				t.Fatalf("messages.3.%s = %q, want step two. Output: %s", tt.wantField, got, string(result))
			}
			if !gjson.GetBytes(result, "messages.3.tool_calls").Exists() {
				t.Fatalf("tool_calls must be preserved. Output: %s", string(result))
			}
			// Non-assistant messages are untouched.
			if !gjson.GetBytes(result, "messages.0.content").Exists() {
				t.Fatalf("user message must be untouched. Output: %s", string(result))
			}
		})
	}
}

func TestApplyReasoningFormatFromHeadersEdgeCases(t *testing.T) {
	t.Run("empty payload returns as-is", func(t *testing.T) {
		in := []byte{}
		got := ApplyReasoningFormatFromHeaders(http.Header{"X-Reasoning-Format": {"openrouter"}}, in)
		if len(got) != 0 {
			t.Fatalf("expected empty payload unchanged, got %s", string(got))
		}
	})
	t.Run("nil headers returns payload unchanged", func(t *testing.T) {
		in := []byte(`{"messages":[{"role":"assistant","reasoning_content":"x"}]}`)
		got := ApplyReasoningFormatFromHeaders(nil, in)
		if string(got) != string(in) {
			t.Fatalf("payload changed without header: %s", string(got))
		}
	})
	t.Run("messages not array returns payload unchanged", func(t *testing.T) {
		in := []byte(`{"messages":"oops"}`)
		got := ApplyReasoningFormatFromHeaders(http.Header{"X-Reasoning-Format": {"openrouter"}}, in)
		if string(got) != string(in) {
			t.Fatalf("payload changed for non-array messages: %s", string(got))
		}
	})
	t.Run("assistant without reasoning_content untouched", func(t *testing.T) {
		in := []byte(`{"messages":[{"role":"assistant","content":"hi"}]}`)
		got := ApplyReasoningFormatFromHeaders(http.Header{"X-Reasoning-Format": {"openrouter"}}, in)
		if gjson.GetBytes(got, "messages.0.reasoning").Exists() {
			t.Fatalf("empty reasoning must not be created. Output: %s", string(got))
		}
	})
}

func TestApplyReasoningFormatToResponseFromHeaders(t *testing.T) {
	nonStream := `{"id":"cmpl-1","choices":[
		{"index":0,"message":{"role":"assistant","content":"answer","reasoning":"or text","reasoning_content":"ds text"}},
		{"index":1,"message":{"role":"assistant","content":"answer2","reasoning":"or2 text","reasoning_content":null}}
	]}`
	streamChunk := `{"id":"chatcmpl-1","choices":[{"index":0,"delta":{"reasoning":"or delta","reasoning_content":"ds delta","content":"answer"}}]}`

	tests := []struct {
		name        string
		headers     http.Header
		wantKept    string // field expected to survive
		wantRemoved string // field expected to be stripped
	}{
		{name: "no header strips reasoning", headers: nil, wantKept: "reasoning_content", wantRemoved: "reasoning"},
		{name: "deepseek strips reasoning", headers: http.Header{"X-Reasoning-Format": {"deepseek"}}, wantKept: "reasoning_content", wantRemoved: "reasoning"},
		{name: "unknown strips reasoning", headers: http.Header{"X-Reasoning-Format": {"weird"}}, wantKept: "reasoning_content", wantRemoved: "reasoning"},
		{name: "openrouter strips reasoning_content", headers: http.Header{"X-Reasoning-Format": {"openrouter"}}, wantKept: "reasoning", wantRemoved: "reasoning_content"},
	}

	for _, tt := range tests {
		t.Run(tt.name+"/non-stream", func(t *testing.T) {
			result := ApplyReasoningFormatToResponseFromHeaders(tt.headers, []byte(nonStream))
			// choice 0 keeps the preferred field with its value intact.
			if got := gjson.GetBytes(result, "choices.0.message."+tt.wantKept).String(); got == "" {
				t.Fatalf("choices.0.message.%s missing or empty. Output: %s", tt.wantKept, string(result))
			}
			if gjson.GetBytes(result, "choices.0.message."+tt.wantRemoved).Exists() {
				t.Fatalf("choices.0.message.%s must be stripped. Output: %s", tt.wantRemoved, string(result))
			}
			// choice 1 exercises the null value: the removed field disappears, kept stays.
			if gjson.GetBytes(result, "choices.1.message."+tt.wantRemoved).Exists() {
				t.Fatalf("choices.1.message.%s must be stripped. Output: %s", tt.wantRemoved, string(result))
			}
			if !gjson.GetBytes(result, "choices.1.message."+tt.wantKept).Exists() {
				t.Fatalf("choices.1.message.%s must survive. Output: %s", tt.wantKept, string(result))
			}
		})
		t.Run(tt.name+"/stream", func(t *testing.T) {
			result := ApplyReasoningFormatToResponseFromHeaders(tt.headers, []byte(streamChunk))
			if got := gjson.GetBytes(result, "choices.0.delta."+tt.wantKept).String(); got == "" {
				t.Fatalf("choices.0.delta.%s missing or empty. Output: %s", tt.wantKept, string(result))
			}
			if gjson.GetBytes(result, "choices.0.delta."+tt.wantRemoved).Exists() {
				t.Fatalf("choices.0.delta.%s must be stripped. Output: %s", tt.wantRemoved, string(result))
			}
			// Unrelated fields must survive.
			if got := gjson.GetBytes(result, "choices.0.delta.content").String(); got != "answer" {
				t.Fatalf("delta.content = %q, want answer. Output: %s", got, string(result))
			}
		})
	}
}

func TestApplyReasoningFormatToResponseFromHeadersEdgeCases(t *testing.T) {
	t.Run("empty payload returns as-is", func(t *testing.T) {
		got := ApplyReasoningFormatToResponseFromHeaders(nil, []byte{})
		if len(got) != 0 {
			t.Fatalf("expected empty payload unchanged, got %s", string(got))
		}
	})
	t.Run("no choices untouched", func(t *testing.T) {
		in := []byte(`{"error":{"message":"boom"}}`)
		got := ApplyReasoningFormatToResponseFromHeaders(nil, in)
		if string(got) != string(in) {
			t.Fatalf("payload without choices changed: %s", string(got))
		}
	})
	t.Run("choices not array untouched", func(t *testing.T) {
		in := []byte(`{"choices":"oops"}`)
		got := ApplyReasoningFormatToResponseFromHeaders(nil, in)
		if string(got) != string(in) {
			t.Fatalf("payload with non-array choices changed: %s", string(got))
		}
	})
}
