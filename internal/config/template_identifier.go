package config

import "strings"

func MatchTemplateIdentifier(id, name, requested string) bool {
	requested = strings.TrimSpace(requested)
	if requested == "" {
		return false
	}

	if strings.EqualFold(strings.TrimSpace(id), requested) || strings.EqualFold(strings.TrimSpace(name), requested) {
		return true
	}

	requestedCanonical := canonicalTemplateIdentifier(requested)
	if requestedCanonical == "" {
		return false
	}

	idCanonical := canonicalTemplateIdentifier(id)
	nameCanonical := canonicalTemplateIdentifier(name)
	if requestedCanonical == idCanonical || requestedCanonical == nameCanonical {
		return true
	}

	if nameCanonical != "" {
		if requestedCanonical == "mig-"+nameCanonical {
			return true
		}
		if strings.TrimPrefix(requestedCanonical, "mig-") == nameCanonical {
			return true
		}
	}

	if idCanonical != "" {
		if requestedCanonical == "mig-"+idCanonical {
			return true
		}
		if strings.TrimPrefix(requestedCanonical, "mig-") == idCanonical {
			return true
		}
	}

	return false
}

func canonicalTemplateIdentifier(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" {
		return ""
	}

	replacer := strings.NewReplacer(
		".", "-",
		"_", "-",
		" ", "-",
		"/", "-",
	)
	value = replacer.Replace(value)
	for strings.Contains(value, "--") {
		value = strings.ReplaceAll(value, "--", "-")
	}
	return strings.Trim(value, "-")
}
