<template>
  <div class="review-manage">
    <div class="page-header">
      <h2>评价管理</h2>
    </div>

    <el-table :data="tableData" border stripe v-loading="loading" style="width: 100%"
      :row-class-name="tableRowClassName">
      <el-table-column prop="id" label="ID" width="70" />
      <el-table-column label="评分" width="170">
        <template #default="{ row }">
          <div class="star-display">
            <span v-for="i in 5" :key="i" class="star" :style="{ color: starColor(row.rating, i) }">★</span>
            <span class="rating-text" :style="{ color: ratingTextColor(row.rating), fontWeight: row.rating <= 2 ? 'bold' : 'normal' }">
              {{ row.rating }}分
              <el-tag v-if="row.rating <= 2" type="danger" size="small" class="warning-tag">差评!</el-tag>
              <el-tag v-else-if="row.rating === 3" type="warning" size="small" class="warning-tag">一般</el-tag>
            </span>
          </div>
        </template>
      </el-table-column>
      <el-table-column prop="content" label="评价内容" min-width="280" show-overflow-tooltip />
      <el-table-column label="时间" width="170">
        <template #default="{ row }">
          {{ row.createTime || '-' }}
        </template>
      </el-table-column>
      <el-table-column prop="isVisible" label="可见" width="80">
        <template #default="{ row }">
          <el-tag :type="row.isVisible ? 'success' : 'info'" size="small">
            {{ row.isVisible ? '可见' : '隐藏' }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="120" fixed="right">
        <template #default="{ row }">
          <el-button
            size="small"
            :type="row.isVisible ? 'warning' : 'success'"
            @click="handleToggleVisibility(row)"
          >
            {{ row.isVisible ? '隐藏' : '显示' }}
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-pagination
      v-model:current-page="currentPage"
      v-model:page-size="pageSize"
      :total="total"
      :page-sizes="[10, 20, 50]"
      layout="total, sizes, prev, pager, next"
      @size-change="loadData"
      @current-change="loadData"
      style="margin-top: 16px; justify-content: flex-end;"
    />
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { getReviews, adminUpdateReviewVisible } from '@/api'

const loading = ref(false)
const tableData = ref([])
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)

// 星级颜色规则：4-5星绿色，3星黄色，1-2星红色
const starColor = (rating, starIndex) => {
  if (starIndex > rating) return '#ddd'
  if (rating <= 2) return '#f56c6c'  // 红色
  if (rating === 3) return '#e6a23c'  // 黄色
  return '#67c23a'  // 绿色
}

const ratingTextColor = (rating) => {
  if (rating <= 2) return '#f56c6c'
  if (rating === 3) return '#e6a23c'
  return '#67c23a'
}

// 差评行红色高亮
const tableRowClassName = ({ row }) => {
  if (row.rating <= 2) return 'warning-row'
  return ''
}

const loadData = async () => {
  loading.value = true
  try {
    const res = await getReviews({ page: currentPage.value, pageSize: pageSize.value })
    tableData.value = res.data?.records || res.data?.list || res.data || []
    total.value = res.data?.total || 0
  } catch (e) {
    ElMessage.error('加载评价列表失败')
  } finally {
    loading.value = false
  }
}

const handleToggleVisibility = async (row) => {
  try {
    await adminUpdateReviewVisible(row.id, row.isVisible ? 0 : 1)
    ElMessage.success(row.isVisible ? '已隐藏' : '已显示')
    loadData()
  } catch (e) {
    ElMessage.error('操作失败')
  }
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.review-manage {
  padding: 10px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.page-header h2 {
  margin: 0;
  font-size: 20px;
  color: #333;
}

.star-display {
  display: flex;
  align-items: center;
}

.star {
  color: #ddd;
  font-size: 18px;
  letter-spacing: 2px;
}

.rating-text {
  margin-left: 8px;
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.warning-tag {
  font-size: 11px;
}

/* 差评行红色背景 */
:deep(.warning-row) {
  background-color: #fef0f0 !important;
}

:deep(.warning-row:hover > td) {
  background-color: #fde2e2 !important;
}
</style>
