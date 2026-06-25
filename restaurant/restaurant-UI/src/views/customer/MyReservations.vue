<template>
  <div class="my-reservations">
    <h1 class="page-title">我的预订</h1>

    <div class="reservation-list" v-loading="loading">
      <el-card
        v-for="item in reservations"
        :key="item.id"
        shadow="hover"
        class="reservation-card"
      >
        <div class="card-header">
          <span class="order-no">订单号: {{ item.orderNo }}</span>
          <el-tag :type="statusTagType(item.status)" size="small">
            {{ statusLabel(item.status) }}
          </el-tag>
        </div>
        <div class="card-body">
          <div class="info-row">
            <span class="label">门店:</span>
            <span class="value">{{ item.storeName }}</span>
          </div>
          <div class="info-row">
            <span class="label">日期:</span>
            <span class="value">{{ item.reserveDate }}</span>
          </div>
          <div class="info-row">
            <span class="label">时段:</span>
            <span class="value">{{ item.timeSlotId ? '已选时段' : '-' }}</span>
          </div>
          <div class="info-row">
            <span class="label">人数:</span>
            <span class="value">{{ item.guestCount }}人</span>
          </div>
        </div>
        <div class="card-footer">
          <el-button
            v-if="canCancel(item.status)"
            type="danger"
            size="small"
            @click="handleCancel(item.id)"
          >
            取消预订
          </el-button>
          <el-button
            v-if="isCompleted(item.status) && !item.hasReviewed"
            type="primary"
            size="small"
            @click="openReviewDialog(item)"
          >
            去评价
          </el-button>
          <span v-if="isCompleted(item.status) && item.hasReviewed" class="reviewed-tag">已评价</span>
        </div>
      </el-card>
      <el-empty v-if="reservations.length === 0 && !loading" description="暂无预订记录" />
    </div>

    <el-pagination
      v-model:current-page="currentPage"
      v-model:page-size="pageSize"
      :total="total"
      :page-sizes="[10, 20]"
      layout="total, prev, pager, next"
      @current-change="loadData"
      style="margin-top: 20px; justify-content: center;"
    />

    <!-- 评价对话框 -->
    <el-dialog v-model="reviewDialogVisible" title="评价" width="450px">
      <el-form :model="reviewForm" label-width="80px">
        <el-form-item label="评分">
          <div class="star-rating">
            <span
              v-for="i in 5"
              :key="i"
              class="rating-star"
              :class="{ active: i <= reviewForm.rating }"
              @click="reviewForm.rating = i"
            >★</span>
          </div>
        </el-form-item>
        <el-form-item label="评价内容">
          <el-input v-model="reviewForm.content" type="textarea" :rows="4" placeholder="请分享您的用餐体验" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="reviewDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="submitReview" :loading="submitting">提交评价</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { getMyReservations, cancelReservation, createReview } from '@/api'

const loading = ref(false)
const reservations = ref([])
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)

const reviewDialogVisible = ref(false)
const submitting = ref(false)
const currentReservation = ref(null)

const reviewForm = ref({
  rating: 5,
  content: ''
})

const statusMap = {
  0: { label: '待确认', type: 'warning' },
  1: { label: '已确认', type: '' },
  2: { label: '已到店', type: 'success' },
  3: { label: '已完成', type: 'info' },
  4: { label: '已取消', type: 'danger' },
  5: { label: '已拒绝', type: 'danger' },
  6: { label: '未到店', type: 'danger' }
}

const statusLabel = (status) => statusMap[status]?.label || `状态${status}`
const statusTagType = (status) => statusMap[status]?.type || 'info'

// 判断是否可以取消（0=待确认, 1=已确认）
const canCancel = (status) => status === 0 || status === 1
// 判断是否已完成（3=已完成）
const isCompleted = (status) => status === 3

const loadData = async () => {
  loading.value = true
  try {
    const res = await getMyReservations({ page: currentPage.value, pageSize: pageSize.value })
    reservations.value = res.data?.records || res.data?.list || res.data || []
    total.value = res.data?.total || 0
  } catch (e) {
    ElMessage.error('加载预订列表失败')
  } finally {
    loading.value = false
  }
}

const handleCancel = (id) => {
  ElMessageBox.confirm('确定要取消此预订吗？', '提示', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'warning'
  }).then(async () => {
    try {
      await cancelReservation(id)
      ElMessage.success('已取消预订')
      loadData()
    } catch (e) {
      ElMessage.error('取消失败')
    }
  }).catch(() => {})
}

const openReviewDialog = (item) => {
  currentReservation.value = item
  reviewForm.value = { rating: 5, content: '' }
  reviewDialogVisible.value = true
}

const submitReview = async () => {
  if (!reviewForm.value.content.trim()) {
    ElMessage.warning('请输入评价内容')
    return
  }
  submitting.value = true
  try {
    await createReview({
      reservationId: currentReservation.value.id,
      storeId: currentReservation.value.storeId,
      rating: reviewForm.value.rating,
      content: reviewForm.value.content
    })
    ElMessage.success('评价成功，感谢您的反馈！')
    reviewDialogVisible.value = false
    loadData()
  } catch (e) {
    ElMessage.error('评价失败')
  } finally {
    submitting.value = false
  }
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.my-reservations {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

.page-title {
  text-align: center;
  margin-bottom: 24px;
  font-size: 24px;
  color: #333;
}

.reservation-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.reservation-card :deep(.el-card__body) {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid #eee;
}

.order-no {
  font-size: 14px;
  color: #666;
  font-weight: bold;
}

.card-body {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.info-row {
  display: flex;
  align-items: center;
  font-size: 14px;
}

.label {
  color: #999;
  margin-right: 8px;
}

.value {
  color: #333;
}

.card-footer {
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid #eee;
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}

.star-rating {
  display: flex;
  gap: 4px;
}

.rating-star {
  font-size: 28px;
  color: #ddd;
  cursor: pointer;
  transition: color 0.2s;
}

.rating-star.active {
  color: #f7ba2a;
}

.rating-star:hover {
  color: #f7ba2a;
}

.reviewed-tag {
  font-size: 12px;
  color: #67c23a;
  border: 1px solid #e1f3d8;
  background: #f0f9eb;
  padding: 2px 10px;
  border-radius: 4px;
}
</style>
