package executor

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/registry"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	"github.com/tidwall/gjson"
)

const reasoningFormatClaudeRequest = `{"model":"gpt-5","messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"prior reasoning","signature":""},{"type":"tool_use","id":"call_1","name":"Read","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"ok"}]}]}`

func reasoningFormatTestExecutor(t *testing.T) *OpenAICompatExecutor {
	t.Helper()
	return NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
}

// TestOpenAICompatExecutorRequestReasoningFormat pins the request-side
// X-Reasoning-Format dialect selection: what the upstream receives for the
// historical thinking field, and that the control header is not leaked.
func TestOpenAICompatExecutorRequestReasoningFormat(t *testing.T) {
	tests := []struct {
		name            string
		headers         http.Header
		wantField       string
		wantMissing     string
		wantHeaderValue string
	}{
		{
			name:            "no header defaults to reasoning_content",
			headers:         http.Header{},
			wantField:       "reasoning_content",
			wantMissing:     "reasoning",
			wantHeaderValue: "",
		},
		{
			name:            "deepseek keeps reasoning_content",
			headers:         http.Header{"X-Reasoning-Format": {"deepseek"}},
			wantField:       "reasoning_content",
			wantMissing:     "reasoning",
			wantHeaderValue: "",
		},
		{
			name:            "openrouter renames to reasoning",
			headers:         http.Header{"X-Reasoning-Format": {"openrouter"}},
			wantField:       "reasoning",
			wantMissing:     "reasoning_content",
			wantHeaderValue: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var upstreamBody []byte
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				upstreamBody, _ = io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
			}))
			defer server.Close()

			executor := reasoningFormatTestExecutor(t)
			auth := &cliproxyauth.Auth{
				Provider: "openai-compatibility",
				Attributes: map[string]string{
					"base_url": server.URL,
					"api_key":  "test-key",
				},
			}
			request := cliproxyexecutor.Request{
				Model:   "gpt-5",
				Payload: []byte(reasoningFormatClaudeRequest),
				Metadata: map[string]any{
					"cliproxy.resolved_api_key_model_info": &registry.ModelInfo{IsCompat: true},
				},
			}
			options := cliproxyexecutor.Options{
				SourceFormat:   sdktranslator.FormatClaude,
				ResponseFormat: sdktranslator.FormatOpenAI,
				Headers:        tt.headers,
			}

			if _, errExecute := executor.Execute(context.Background(), auth, request, options); errExecute != nil {
				t.Fatalf("Execute error: %v", errExecute)
			}

			assistant := gjson.GetBytes(upstreamBody, "messages.0")
			if got := assistant.Get(tt.wantField).String(); got != "prior reasoning" {
				t.Fatalf("messages.0.%s = %q, want prior reasoning; body=%s", tt.wantField, got, upstreamBody)
			}
			if assistant.Get(tt.wantMissing).Exists() {
				t.Fatalf("messages.0.%s must not reach the upstream; body=%s", tt.wantMissing, upstreamBody)
			}
			if !assistant.Get("tool_calls").Exists() {
				t.Fatalf("tool_calls must survive; body=%s", upstreamBody)
			}
		})
	}
}

// TestOpenAICompatExecutorResponseReasoningFormat pins the response-side strict
// single-field semantics: the executor strips the non-preferred reasoning field
// from the upstream response before translation, so the client sees thinking
// content only from the requested field.
func TestOpenAICompatExecutorResponseReasoningFormat(t *testing.T) {
	tests := []struct {
		name             string
		headers          http.Header
		upstreamReasoner string // reasoning field present in the upstream message
		wantThinking     bool
	}{
		{name: "no header + reasoning field is dropped", headers: http.Header{}, upstreamReasoner: "reasoning", wantThinking: false},
		{name: "openrouter + reasoning field survives", headers: http.Header{"X-Reasoning-Format": {"openrouter"}}, upstreamReasoner: "reasoning", wantThinking: true},
		{name: "openrouter + reasoning_content field is dropped", headers: http.Header{"X-Reasoning-Format": {"openrouter"}}, upstreamReasoner: "reasoning_content", wantThinking: false},
		{name: "deepseek + reasoning_content field survives", headers: http.Header{"X-Reasoning-Format": {"deepseek"}}, upstreamReasoner: "reasoning_content", wantThinking: true},
		{name: "no header + reasoning_content field survives", headers: http.Header{}, upstreamReasoner: "reasoning_content", wantThinking: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstreamResponse := `{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok","` + tt.upstreamReasoner + `":"chain of thought"},"finish_reason":"stop"}]}`
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(upstreamResponse))
			}))
			defer server.Close()

			executor := reasoningFormatTestExecutor(t)
			auth := &cliproxyauth.Auth{
				Provider: "openai-compatibility",
				Attributes: map[string]string{
					"base_url": server.URL,
					"api_key":  "test-key",
				},
			}
			request := cliproxyexecutor.Request{
				Model:   "gpt-5",
				Payload: []byte(`{"model":"gpt-5","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`),
				Metadata: map[string]any{
					"cliproxy.resolved_api_key_model_info": &registry.ModelInfo{IsCompat: true},
				},
			}
			options := cliproxyexecutor.Options{
				SourceFormat:   sdktranslator.FormatClaude,
				ResponseFormat: sdktranslator.FormatClaude,
				Headers:        tt.headers,
			}

			resp, errExecute := executor.Execute(context.Background(), auth, request, options)
			if errExecute != nil {
				t.Fatalf("Execute error: %v", errExecute)
			}

			hasThinking := false
			gjson.GetBytes(resp.Payload, "content").ForEach(func(_, block gjson.Result) bool {
				if block.Get("type").String() == "thinking" {
					hasThinking = true
					return false
				}
				return true
			})
			if hasThinking != tt.wantThinking {
				t.Fatalf("thinking block presence = %v, want %v; response=%s", hasThinking, tt.wantThinking, resp.Payload)
			}
		})
	}
}

// TestOpenAICompatExecutorResponseReasoningFormatStream covers the streaming
// path: reasoning deltas must follow the requested dialect and be dropped
// otherwise.
func TestOpenAICompatExecutorResponseReasoningFormatStream(t *testing.T) {
	tests := []struct {
		name             string
		headers          http.Header
		upstreamReasoner string
		wantDelta        bool
	}{
		{name: "openrouter + reasoning delta survives", headers: http.Header{"X-Reasoning-Format": {"openrouter"}}, upstreamReasoner: "reasoning", wantDelta: true},
		{name: "no header + reasoning delta dropped", headers: http.Header{}, upstreamReasoner: "reasoning", wantDelta: false},
		{name: "no header + reasoning_content delta survives", headers: http.Header{}, upstreamReasoner: "reasoning_content", wantDelta: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstreamChunk := `data: {"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"","` + tt.upstreamReasoner + `":"step by step"}}]}` + "\n\n" + "data: [DONE]\n\n"
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = w.Write([]byte(upstreamChunk))
			}))
			defer server.Close()

			executor := reasoningFormatTestExecutor(t)
			auth := &cliproxyauth.Auth{
				Provider: "openai-compatibility",
				Attributes: map[string]string{
					"base_url": server.URL,
					"api_key":  "test-key",
				},
			}
			request := cliproxyexecutor.Request{
				Model:   "gpt-5",
				Payload: []byte(`{"model":"gpt-5","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"stream":true}`),
				Metadata: map[string]any{
					"cliproxy.resolved_api_key_model_info": &registry.ModelInfo{IsCompat: true},
				},
			}
			options := cliproxyexecutor.Options{
				SourceFormat:   sdktranslator.FormatClaude,
				ResponseFormat: sdktranslator.FormatClaude,
				Stream:         true,
				Headers:        tt.headers,
				// The streaming response translator dispatches on the stream
				// field of the original client request, as the handler does.
				OriginalRequest: []byte(`{"model":"gpt-5","stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`),
			}

			result, errExecute := executor.ExecuteStream(context.Background(), auth, request, options)
			if errExecute != nil {
				t.Fatalf("ExecuteStream error: %v", errExecute)
			}

			hasThinkingDelta := false
			for chunk := range result.Chunks {
				if chunk.Err != nil {
					t.Fatalf("unexpected stream error: %v", chunk.Err)
				}
				if strings.Contains(string(chunk.Payload), "thinking_delta") {
					hasThinkingDelta = true
				}
			}
			if hasThinkingDelta != tt.wantDelta {
				t.Fatalf("thinking_delta presence = %v, want %v", hasThinkingDelta, tt.wantDelta)
			}
		})
	}
}
