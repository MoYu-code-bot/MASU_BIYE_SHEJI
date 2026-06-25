<template>
  <div class="review-manage">
    <div class="page-header">
      <h2>评价管理</h2>
    </div>

    <el-table :data="tableData" border stripe v-loading="loading" style="width: 100%">
      <el-table-column prop="id" label="ID" width="70" />
      <el-table-column prop="storeName" label="门店" min-width="120" />
      <el-table-column prop="rating" label="评分" width="160">
        <template #default="{ row }">
          <div class="star-display">
            <span v-for="i in 5" :key="i" class="star" :class="{ active: i <= row.rating }">★</span>
            <span class="rating-text">{{ row.rating }}分</span>
          </div>
        </template>
      </el-table-column>
      <el-table-column prop="content" label="评价内容" min-width="250" show-overflow-tooltip />
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
import { getReviews, updateReview } from '@/api'

const loading = ref(false)
const tableData = ref([])
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)

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
    await updateReview(row.id, { ...row, isVisible: !row.isVisible })
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
  font-size: 16px;
}

.star.active {
  color: #f7ba2a;
}

.rating-text {
  margin-left: 6px;
  font-size: 13px;
  color: #666;
}
</style>
