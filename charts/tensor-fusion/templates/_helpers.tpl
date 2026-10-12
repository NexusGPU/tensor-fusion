{{/*
Expand the name of the chart.
*/}}
{{- define "tensor-fusion.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "tensor-fusion.namespace" -}}
{{- default .Release.Namespace .Values.namespaceOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).

Precedence: fullnameOverride > release name (with nameOverride fallback when
the release name does not contain the chart/name token). This matches the
standard Helm/Bitnami pattern so users setting only fullnameOverride see
their value reflected in every resource name.
*/}}
{{- define "tensor-fusion.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- default .Release.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "tensor-fusion.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "tensor-fusion.labels" -}}
helm.sh/chart: {{ include "tensor-fusion.chart" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "tensor-fusion.controller.selectorLabels" -}}
app.kubernetes.io/name: {{ include "tensor-fusion.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: controller
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "tensor-fusion.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "tensor-fusion.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
The cluster agent's credentials (enroll token and agent id) are never rendered
into a pod spec: they live in a Secret and are injected by reference. The
Secret is either created by the chart from agent.enrollToken / agent.agentId,
or managed by the user and named in agent.existingSecret.

agent.existingSecret is read through "default dict" everywhere: a release
upgraded with --reuse-values carries the values of the chart version it was
installed from, in which the key does not exist.
*/}}
{{- define "tensor-fusion.agent.enabled" -}}
{{- $existing := .Values.agent.existingSecret | default dict -}}
{{- if or .Values.agent.agentId (and $existing.name $existing.agentIdKey) -}}true{{- end -}}
{{- end }}

{{/* Set when the chart creates the agent Secret itself. */}}
{{- define "tensor-fusion.agent.createSecret" -}}
{{- $existing := .Values.agent.existingSecret | default dict -}}
{{- if and .Values.agent.agentId (not $existing.name) -}}true{{- end -}}
{{- end }}

{{- define "tensor-fusion.agent.secretName" -}}
{{- $existing := .Values.agent.existingSecret | default dict -}}
{{- $existing.name | default (printf "%s-agent" (include "tensor-fusion.fullname" .)) -}}
{{- end }}

{{- define "tensor-fusion.agent.tokenKey" -}}
{{- $existing := .Values.agent.existingSecret | default dict -}}
{{- if $existing.name -}}{{ $existing.key | default "enroll-token" }}{{- else -}}enroll-token{{- end -}}
{{- end }}

{{/* Key of the agent id in the Secret; empty when it is taken from agent.agentId. */}}
{{- define "tensor-fusion.agent.idKey" -}}
{{- $existing := .Values.agent.existingSecret | default dict -}}
{{- if $existing.name -}}{{ $existing.agentIdKey | default "" }}{{- else -}}agent-id{{- end -}}
{{- end }}

{{/*
A namespace list of the cluster agent (agent.namespaceAllowlist or
agent.namespaceDenylist) as the comma-separated value the agent reads. Every
entry must be a namespace name, or a prefix followed by "*". The agent exits at
start on anything else, and it runs in the controller's pod, so a mistake is
refused here, when the chart is rendered.
Call with (list "<value name>" <the list>).
*/}}
{{- define "tensor-fusion.agent.namespaceList" -}}
{{- $name := index . 0 -}}
{{- $entries := index . 1 | default list -}}
{{- if not (kindIs "slice" $entries) -}}
{{- fail (printf "agent.%s must be a list of namespace names" $name) -}}
{{- end -}}
{{- range $entries -}}
{{- if not (kindIs "string" .) -}}
{{- fail (printf "agent.%s: every entry must be a string, got %v" $name .) -}}
{{- end -}}
{{- if not (regexMatch "^([a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?|[a-z0-9][-a-z0-9]{0,61}[*])$" .) -}}
{{- fail (printf "agent.%s: %q is not a namespace name, nor a prefix followed by *" $name .) -}}
{{- end -}}
{{- end -}}
{{- join "," $entries -}}
{{- end }}
