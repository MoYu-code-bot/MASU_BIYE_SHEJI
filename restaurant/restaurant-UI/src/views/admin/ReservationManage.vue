<template>
  <div class="reservation-manage">
    <div class="page-header">
      <h2>预订管理</h2>
    </div>

    <!-- 筛选 -->
    <div class="filter-bar">
      <el-select v-model="filterStatus" placeholder="按状态筛选" clearable @change="loadData" style="width: 150px; margin-right: 12px;">
        <el-option label="全部状态" :value="null" />
        <el-option label="待确认" value="PENDING" />
        <el-option label="已确认" value="CONFIRMED" />
        <el-option label="已到店" value="ARRIVED" />
        <el-option label="已完成" value="COMPLETED" />
        <el-option label="已取消" value="CANCELLED" />
        <el-option label="已拒绝" value="REJECTED" />
        <el-option label="未到店" value="NO_SHOW" />
      </el-select>
      <el-date-picker
        v-model="filterDate"
        type="date"
        placeholder="选择日期"
        format="YYYY-MM-DD"
        value-format="YYYY-MM-DD"
        @change="loadData"
        style="width: 180px;"
      />
    </div>

    <el-table :data="tableData" border stripe v-loading="loading" style="width: 100%">
      <el-table-column prop="orderNo" label="订单号" width="180" />
      <el-table-column prop="storeName" label="门店" min-width="120" />
      <el-table-column prop="reserveDate" label="预订日期" width="120" />
      <el-table-column prop="timeSlot" label="时段" width="120" />
      <el-table-column prop="guestCount" label="人数" width="70" />
      <el-table-column prop="customerName" label="预订人" width="100" />
      <el-table-column prop="customerPhone" label="手机" width="130" />
      <el-table-column prop="status" label="状态" width="100">
        <template #default="{ row }">
          <el-tag :type="statusTagType(row.status)">
            {{ statusLabel(row.status) }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="260" fixed="right">
        <template #default="{ row }">
          <el-button
            v-if="row.status === 'PENDING'"
            size="small" type="success"
            @click="handleAction(row.id, 'CONFIRMED')"
          >确认</el-button>
          <el-button
            v-if="row.status === 'PENDING'"
            size="small" type="danger"
            @click="handleAction(row.id, 'REJECTED')"
          >拒绝</el-button>
          <el-button
            v-if="row.status === 'CONFIRMED'"
            size="small" type="primary"
            @click="handleAction(row.id, 'ARRIVED')"
          >到店</el-button>
          <el-button
            v-if="row.status === 'ARRIVED'"
            size="small" type="success"
            @click="handleAction(row.id, 'COMPLETED')"
          >完成</el-button>
          <el-button
            v-if="row.status === 'CONFIRMED'"
            size="small" type="warning"
            @click="handleAction(row.id, 'NO_SHOW')"
          >未到店</el-button>
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
import { ElMessage, ElMessageBox } from 'element-plus'
import { getReservations, updateReservationStatus } from '@/api'

const loading = ref(false)
const tableData = ref([])
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)
const filterStatus = ref(null)
const filterDate = ref(null)

const statusMap = {
  PENDING: { label: '待确认', type: 'warning' },
  CONFIRMED: { label: '已确认', type: '' },
  ARRIVED: { label: '已到店', type: 'success' },
  COMPLETED: { label: '已完成', type: 'info' },
  CANCELLED: { label: '已取消', type: 'danger' },
  REJECTED: { label: '已拒绝', type: 'danger' },
  NO_SHOW: { label: '未到店', type: 'danger' }
}

const statusLabel = (status) => {
  return statusMap[status]?.label || status
}

const statusTagType = (status) => {
  return statusMap[status]?.type || 'info'
}

const loadData = async () => {
  loading.value = true
  try {
    const params = {
      page: currentPage.value,
      pageSize: pageSize.value
    }
    if (filterStatus.value) params.status = filterStatus.value
    if (filterDate.value) params.reserveDate = filterDate.value

    const res = await getReservations(params)
    tableData.value = res.data?.records || res.data?.list || res.data || []
    total.value = res.data?.total || 0
  } catch (e) {
    ElMessage.error('加载预订列表失败')
  } finally {
    loading.value = false
  }
}

const handleAction = (id, newStatus) => {
  const actionLabel = statusLabel(newStatus)
  ElMessageBox.confirm(`确定将此预订状态改为「${actionLabel}」吗？`, '操作确认', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'warning'
  }).then(async () => {
    try {
      await updateReservationStatus(id, newStatus)
      ElMessage.success(`已${actionLabel}`)
      loadData()
    } catch (e) {
      ElMessage.error('操作失败')
    }
  }).catch(() => {})
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.reservation-manage {
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

.filter-bar {
  margin-bottom: 16px;
  display: flex;
  align-items: center;
}
</style>
