/*
 * Simple sanity tests for Ascend accelerator implementation (fake mode).
 * Validates device enumeration and partition assign/remove behaviours.
 */

#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "../accelerator.h"

static void test_device_info() {
    size_t count = 0;
    Result r = GetDeviceCount(&count);
    assert(r == RESULT_SUCCESS);
    assert(count > 0);

    ExtendedDeviceInfo infos[4];
    memset(infos, 0, sizeof(infos));
    size_t returned = 0;
    r = GetAllDevices(infos, 4, &returned);
    assert(r == RESULT_SUCCESS);
    assert(returned == count || returned == 4); /* capped by buffer size */

    for (size_t i = 0; i < returned; i++) {
        assert(strlen(infos[i].basic.uuid) > 0);
        assert(infos[i].capabilities.supportsPartitioning == true);
    }
}

static void test_partition_templates() {
    PartitionTemplate templates[8];
    size_t tmplCount = 0;
    Result r = GetPartitionTemplates(0, templates, 8, &tmplCount);
    assert(r == RESULT_SUCCESS);
    assert(tmplCount > 0);
    for (size_t i = 0; i < tmplCount; i++) {
        assert(strlen(templates[i].templateId) > 0);
        assert(templates[i].memoryBytes > 0);
    }
}

static void test_partition_lifecycle() {
    PartitionAssignment assignment;
    memset(&assignment, 0, sizeof(assignment));
    snprintf(assignment.templateId, sizeof(assignment.templateId), "vir01");
    snprintf(assignment.deviceUUID, sizeof(assignment.deviceUUID), "npu-0-chip-0");

    bool ok = AssignPartition(&assignment);
    assert(ok == true);
    assert(strlen(assignment.partitionUUID) > 0);
    assert(assignment.partitionOverheadBytes > 0);

    ok = RemovePartition("vir01", "npu-0-chip-0");
    assert(ok == true);

    /* invalid args */
    PartitionAssignment invalid;
    memset(&invalid, 0, sizeof(invalid));
    ok = AssignPartition(&invalid);
    assert(ok == false);
    ok = RemovePartition(NULL, NULL);
    assert(ok == false);
}

int main() {
    size_t count = 0;
    Result rc = GetDeviceCount(&count);
    if (rc != RESULT_SUCCESS || count == 0) {
        printf("No Ascend devices detected; skipping vNPU checks.\n");
        return 0;
    }

    test_device_info();
    test_partition_templates();
    test_partition_lifecycle();
    printf("Ascend fake partition tests passed.\n");
    return 0;
}
