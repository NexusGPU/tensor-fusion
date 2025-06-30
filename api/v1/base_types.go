package v1

import "fmt"

type NameNamespace struct {
	Name      string `json:"name,omitempty"`
	Namespace string `json:"namespace,omitempty"`
}

func (n NameNamespace) String() string {
	return fmt.Sprintf("%s/%s", n.Namespace, n.Name)
}
