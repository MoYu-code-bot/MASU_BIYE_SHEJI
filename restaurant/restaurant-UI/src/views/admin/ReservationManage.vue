<template>
  <div class="reservation-manage">
    <div class="page-header">
      <h2>预订管理</h2>
    </div>

    <!-- 筛选 -->
    <div class="filter-bar">
      <el-select v-model="filterStatus" placeholder="按状态筛选" clearable @change="loadData" style="width: 150px; margin-right: 12px;">
        <el-option label="全部状态" :value="null" />
        <el-option label="待确认" :value="0" />
        <el-option label="已确认" :value="1" />
        <el-option label="已到店" :value="2" />
        <el-option label="已完成" :value="3" />
        <el-option label="已取消" :value="4" />
        <el-option label="已拒绝" :value="5" />
        <el-option label="未到店" :value="6" />
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
      <el-table-column prop="reserveDate" label="预订日期" width="120" />
      <el-table-column prop="guestCount" label="人数" width="70" />
      <el-table-column prop="customerName" label="预订人" width="100" />
      <el-table-column prop="customerPhone" label="手机" width="130" />
      <el-table-column label="状态" width="100">
        <template #default="{ row }">
          <el-tag :type="statusTagType(row.status)">
            {{ statusLabel(row.status) }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="280" fixed="right">
        <template #default="{ row }">
          <el-button
            v-if="row.status === 0"
            size="small" type="success"
            @click="handleAction(row.id, 'confirm')"
          >确认</el-button>
          <el-button
            v-if="row.status === 0"
            size="small" type="danger"
            @click="handleAction(row.id, 'reject')"
          >拒绝</el-button>
          <el-button
            v-if="row.status === 1"
            size="small" type="primary"
            @click="handleAction(row.id, 'arrive')"
          >到店</el-button>
          <el-button
            v-if="row.status === 2"
            size="small" type="success"
            @click="handleAction(row.id, 'complete')"
          >完成</el-button>
          <el-button
            v-if="row.status === 1"
            size="small" type="warning"
            @click="handleAction(row.id, 'noshow')"
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

// 状态码 → { label, type }
const statusMap = {
  0: { label: '待确认', type: 'warning' },
  1: { label: '已确认', type: '' },
  2: { label: '已到店', type: 'success' },
  3: { label: '已完成', type: 'info' },
  4: { label: '已取消', type: 'danger' },
  5: { label: '已拒绝', type: 'danger' },
  6: { label: '未到店', type: 'danger' }
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
    if (filterStatus.value !== null) params.status = filterStatus.value
    if (filterDate.value) params.date = filterDate.value

    const res = await getReservations(params)
    tableData.value = res.data?.records || res.data?.list || res.data || []
    total.value = res.data?.total || 0
  } catch (e) {
    ElMessage.error('加载预订列表失败')
  } finally {
    loading.value = false
  }
}

const actionLabels = {
  confirm: '确认',
  reject: '拒绝',
  arrive: '到店',
  complete: '完成',
  noshow: '未到店'
}

const handleAction = (id, action) => {
  const label = actionLabels[action] || action
  ElMessageBox.confirm(`确定将此预订状态改为「${label}」吗？`, '操作确认', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'warning'
  }).then(async () => {
    try {
      await updateReservationStatus(id, action)
      ElMessage.success(`已${label}`)
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
