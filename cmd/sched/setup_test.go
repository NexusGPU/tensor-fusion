/*
Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0
*/

package sched

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/component-base/configz"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/latest"
)

func TestSched(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Sched Suite")
}

// These specs guard against regressions when bumping k8s.io/component-base, which
// since v0.36 requires runtime objects passed to configz.Set to declare a GVK and
// an external (non-internal) version. Without the conversion + GVK setup the
// scheduler used to crash at boot with "val must specify a kind".
var _ = Describe("ToExternalSchedulerConfig", func() {
	It("converts the default scheduler config and registers it through configz", func() {
		// configz keeps a process-wide registry; isolate this spec's entry.
		const cfgName = "tf-scheduler-test-default"
		defer configz.Delete(cfgName)

		internal, err := latest.Default()
		Expect(err).NotTo(HaveOccurred())

		external, err := ToExternalSchedulerConfig(internal)
		Expect(err).NotTo(HaveOccurred())
		Expect(external).NotTo(BeNil())

		gvk := external.GroupVersionKind()
		Expect(gvk.Kind).To(Equal("KubeSchedulerConfiguration"))
		Expect(gvk.Version).To(Equal("v1"))
		Expect(gvk.Group).To(Equal("kubescheduler.config.k8s.io"))

		cz, err := configz.New(cfgName)
		Expect(err).NotTo(HaveOccurred())
		Expect(cz.Set(external)).To(Succeed(),
			"configz.Set must accept the converted external config; "+
				"check k8s.io/component-base/configz GVK validation when this fails")
	})

	It("preserves the configured profile names from the internal config", func() {
		internal, err := latest.Default()
		Expect(err).NotTo(HaveOccurred())
		Expect(internal.Profiles).NotTo(BeEmpty())

		external, err := ToExternalSchedulerConfig(internal)
		Expect(err).NotTo(HaveOccurred())
		Expect(external.Profiles).To(HaveLen(len(internal.Profiles)))
		for i := range internal.Profiles {
			Expect(external.Profiles[i].SchedulerName).NotTo(BeNil())
			Expect(*external.Profiles[i].SchedulerName).To(Equal(internal.Profiles[i].SchedulerName))
		}
	})
})
