#include <gtest/gtest.h>
#include "../../accelerator.h"
#include <string>
#include <vector>

TEST(AscendProviderTest, GetDeviceCount) {
    size_t count = 0;
    Result res = GetDeviceCount(&count);
    if (res != RESULT_SUCCESS || count == 0) {
        GTEST_SKIP() << "No Ascend devices detected via dcmi";
    }
    ASSERT_EQ(res, RESULT_SUCCESS);
}

TEST(AscendProviderTest, GetAllDevices) {
    size_t count = 0;
    GetDeviceCount(&count);
    if (count == 0) {
        GTEST_SKIP() << "No Ascend devices detected via dcmi";
    }
    
    std::vector<ExtendedDeviceInfo> devices(count);
    size_t actualCount = 0;
    Result res = GetAllDevices(devices.data(), count, &actualCount);
    
    EXPECT_EQ(res, RESULT_SUCCESS);
    EXPECT_EQ(actualCount, count);
    EXPECT_STREQ(devices[0].basic.vendor, "Huawei");
    EXPECT_TRUE(devices[0].capabilities.supportsPartitioning);
}

TEST(AscendProviderTest, GetPartitionTemplates) {
    size_t count = 0;
    GetDeviceCount(&count);
    if (count == 0) {
        GTEST_SKIP() << "No Ascend devices detected via dcmi";
    }

    PartitionTemplate templates[8] = {};
    size_t tmplCount = 0;
    Result res = GetPartitionTemplates(0, templates, 8, &tmplCount);
    EXPECT_EQ(res, RESULT_SUCCESS);
    EXPECT_GT(tmplCount, 0);
    EXPECT_STREQ(templates[0].templateId, "vir01");
}

TEST(AscendProviderTest, PartitionLifecycleIdempotent) {
    // Pull a valid template and device
    size_t devCount = 0;
    GetDeviceCount(&devCount);
    if (devCount == 0) {
        GTEST_SKIP() << "No Ascend devices detected via dcmi";
    }
    std::vector<ExtendedDeviceInfo> devices(devCount);
    size_t actualDevCount = 0;
    GetAllDevices(devices.data(), devCount, &actualDevCount);
    ASSERT_GT(actualDevCount, 0);

    PartitionTemplate templates[8] = {};
    size_t tmplCount = 0;
    ASSERT_EQ(GetPartitionTemplates(0, templates, 8, &tmplCount), RESULT_SUCCESS);
    ASSERT_GT(tmplCount, 0);
    const char* templateId = templates[0].templateId;

    // Assign Partition
    PartitionAssignment assignment{};
    snprintf(assignment.templateId, sizeof(assignment.templateId), "%s", templateId);
    snprintf(assignment.deviceUUID, sizeof(assignment.deviceUUID), "%s", devices[0].basic.uuid);
    
    bool success = AssignPartition(&assignment);
    EXPECT_TRUE(success);
    EXPECT_GT(strlen(assignment.partitionUUID), 0);
    EXPECT_GE(assignment.partitionOverheadBytes, 0);

    // Re-assign should be idempotent and reuse UUID
    PartitionAssignment second{};
    snprintf(second.templateId, sizeof(second.templateId), "%s", templateId);
    snprintf(second.deviceUUID, sizeof(second.deviceUUID), "%s", devices[0].basic.uuid);
    success = AssignPartition(&second);
    EXPECT_TRUE(success);
    EXPECT_STREQ(assignment.partitionUUID, second.partitionUUID);
    
    // Remove Partition
    success = RemovePartition(templateId, devices[0].basic.uuid);
    EXPECT_TRUE(success);
}

TEST(AscendProviderTest, InvalidPartitionRemoval) {
    bool success = RemovePartition("invalid-template", "invalid-uuid");
    EXPECT_TRUE(success);
}
