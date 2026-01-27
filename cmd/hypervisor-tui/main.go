/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"context"
	"flag"
	"os"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/tui"
	tea "github.com/charmbracelet/bubbletea"
	"k8s.io/klog/v2"
)

var (
	host = flag.String("host", "localhost", "Hypervisor server host")
	port = flag.Int("port", 8001, "Hypervisor server port")
)

func main() {
	flag.Parse()
	klog.InitFlags(nil)
	defer klog.Flush()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create HTTP client
	client := tui.NewClient(*host, *port)

	// Create TUI model
	model := tui.NewModel(ctx, client)

	// Start TUI
	p := tea.NewProgram(model, tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		klog.Fatalf("Error running TUI: %v", err)
		os.Exit(1)
	}
}
