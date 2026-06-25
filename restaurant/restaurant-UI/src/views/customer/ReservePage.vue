<template>
  <div class="reserve-page">
    <h1 class="page-title">预订门店</h1>

    <!-- 步骤条 -->
    <el-steps :active="currentStep" finish-status="success" class="reserve-steps">
      <el-step title="选择日期" />
      <el-step title="选择时段" />
      <el-step title="填写信息" />
      <el-step title="确认预订" />
    </el-steps>

    <!-- 步骤1: 选择日期 -->
    <div v-if="currentStep === 0" class="step-content">
      <h3>请选择预订日期</h3>
      <el-date-picker
        v-model="selectedDate"
        type="date"
        placeholder="选择日期"
        format="YYYY-MM-DD"
        value-format="YYYY-MM-DD"
        :disabled-date="disablePastDate"
        size="large"
        style="width: 300px;"
      />
      <div class="step-actions">
        <el-button type="primary" @click="nextStep" :disabled="!selectedDate">下一步</el-button>
      </div>
    </div>

    <!-- 步骤2: 选择时段 -->
    <div v-if="currentStep === 1" class="step-content">
      <h3>请选择时段 ({{ selectedDate }})</h3>
      <div class="time-slot-grid">
        <div
          v-for="slot in timeSlots"
          :key="slot.time"
          class="time-slot"
          :class="{ selected: selectedTimeSlot === slot.time, disabled: slot.remaining <= 0 }"
          @click="selectTimeSlot(slot)"
        >
          <span class="slot-time">{{ slot.time }}</span>
          <span class="slot-remaining" :class="{ 'low-stock': slot.remaining <= 3 && slot.remaining > 0 }">
            {{ slot.remaining > 0 ? `剩余${slot.remaining}桌` : '已满' }}
          </span>
        </div>
      </div>
      <el-empty v-if="timeSlots.length === 0" description="该日期暂无可用时段" />
      <div class="step-actions">
        <el-button @click="currentStep--">上一步</el-button>
        <el-button type="primary" @click="nextStep" :disabled="!selectedTimeSlot">下一步</el-button>
      </div>
    </div>

    <!-- 步骤3: 填写信息 -->
    <div v-if="currentStep === 2" class="step-content">
      <h3>请填写预订信息</h3>
      <el-form :model="form" :rules="rules" ref="formRef" label-width="100px" style="max-width: 500px;">
        <el-form-item label="用餐人数" prop="guestCount">
          <el-input-number v-model="form.guestCount" :min="1" :max="20" size="large" />
        </el-form-item>
        <el-form-item label="预订人" prop="customerName">
          <el-input v-model="form.customerName" placeholder="请输入姓名" size="large" />
        </el-form-item>
        <el-form-item label="手机号" prop="customerPhone">
          <el-input v-model="form.customerPhone" placeholder="请输入手机号" size="large" />
        </el-form-item>
        <el-form-item label="备注" prop="remark">
          <el-input v-model="form.remark" type="textarea" :rows="3" placeholder="如有特殊需求请在此备注" />
        </el-form-item>
      </el-form>
      <div class="step-actions">
        <el-button @click="currentStep--">上一步</el-button>
        <el-button type="primary" @click="nextStep">下一步</el-button>
      </div>
    </div>

    <!-- 步骤4: 确认预订 -->
    <div v-if="currentStep === 3" class="step-content">
      <h3>请确认预订信息</h3>
      <el-descriptions :column="1" border class="confirm-info">
        <el-descriptions-item label="预订日期">{{ selectedDate }}</el-descriptions-item>
        <el-descriptions-item label="预订时段">{{ selectedTimeSlot }}</el-descriptions-item>
        <el-descriptions-item label="用餐人数">{{ form.guestCount }}人</el-descriptions-item>
        <el-descriptions-item label="预订人">{{ form.customerName }}</el-descriptions-item>
        <el-descriptions-item label="手机号">{{ form.customerPhone }}</el-descriptions-item>
        <el-descriptions-item label="备注" v-if="form.remark">{{ form.remark }}</el-descriptions-item>
      </el-descriptions>
      <div class="step-actions">
        <el-button @click="currentStep--">上一步</el-button>
        <el-button type="primary" @click="handleSubmit" :loading="submitting">确认提交</el-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { getTimeSlots, createReservation } from '@/api'

const route = useRoute()
const router = useRouter()

const currentStep = ref(0)
const selectedDate = ref('')
const selectedTimeSlot = ref('')
const selectedSlotId = ref(null)
const timeSlots = ref([])
const submitting = ref(false)
const formRef = ref(null)

const form = ref({
  guestCount: 2,
  customerName: '',
  customerPhone: '',
  remark: ''
})

const rules = {
  guestCount: [{ required: true, message: '请选择用餐人数', trigger: 'change' }],
  customerName: [{ required: true, message: '请输入预订人姓名', trigger: 'blur' }],
  customerPhone: [
    { required: true, message: '请输入手机号', trigger: 'blur' },
    { pattern: /^1[3-9]\d{9}$/, message: '手机号格式不正确', trigger: 'blur' }
  ]
}

const disablePastDate = (time) => {
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  return time.getTime() < today.getTime()
}

const loadTimeSlots = async () => {
  if (!selectedDate.value) return
  try {
    const storeId = Number(route.params.storeId)
    const res = await getTimeSlots(storeId)
    // 后端返回 TimeSlot 数组: { id, period, startTime, endTime, maxTables, ... }
    const raw = res.data || res || []
    timeSlots.value = raw.map(s => ({
      id: s.id,
      time: `${s.startTime}-${s.endTime}`,  // 具体时间如 "11:00-12:00"
      remaining: s.maxTables   // 用 maxTables 作为余量展示（默认全部可用）
    }))
  } catch (e) {
    // 如果没有API，使用默认时段
    timeSlots.value = [
      { id: 1, time: '11:00-12:00', remaining: 5 },
      { id: 2, time: '12:00-13:00', remaining: 3 },
      { id: 3, time: '13:00-14:00', remaining: 8 },
      { id: 4, time: '17:00-18:00', remaining: 6 },
      { id: 5, time: '18:00-19:00', remaining: 2 },
      { id: 6, time: '19:00-20:00', remaining: 4 },
      { id: 7, time: '20:00-21:00', remaining: 7 }
    ]
  }
}

const selectTimeSlot = (slot) => {
  if (slot.remaining <= 0) return
  selectedTimeSlot.value = slot.time
  selectedSlotId.value = slot.id
}

const nextStep = () => {
  if (currentStep.value === 0) {
    loadTimeSlots()
    selectedTimeSlot.value = ''
    selectedSlotId.value = null
  }
  if (currentStep.value === 2) {
    formRef.value.validate((valid) => {
      if (valid) currentStep.value++
    })
    return
  }
  currentStep.value++
}

const handleSubmit = async () => {
  submitting.value = true
  try {
    const storeId = route.params.storeId
    const gc = form.value.guestCount
    // 根据用餐人数自动匹配桌型：1=2人桌 2=4人桌 3=6人桌 4=8人桌 5=大包间
    const tableTypeId = gc <= 2 ? 1 : gc <= 4 ? 2 : gc <= 6 ? 3 : gc <= 8 ? 4 : 5
    await createReservation({
      storeId: Number(storeId),
      reserveDate: selectedDate.value,
      timeSlotId: selectedSlotId.value,
      tableTypeId: tableTypeId,
      guestCount: gc,
      customerName: form.value.customerName,
      customerPhone: form.value.customerPhone,
      remark: form.value.remark
    })
    ElMessage.success('预订成功！')
    router.push('/c/my-reservations')
  } catch (e) {
    ElMessage.error('预订失败，请稍后重试')
  } finally {
    submitting.value = false
  }
}

onMounted(() => {
  // 预填手机号
  const phone = localStorage.getItem('customerPhone')
  if (phone) {
    form.value.customerPhone = phone
  }
})
</script>

<style scoped>
.reserve-page {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

.page-title {
  text-align: center;
  margin-bottom: 30px;
  font-size: 24px;
  color: #333;
}

.reserve-steps {
  margin-bottom: 40px;
}

.step-content {
  text-align: center;
  padding: 20px;
}

.step-content h3 {
  margin-bottom: 24px;
  font-size: 18px;
  color: #333;
}

.step-actions {
  margin-top: 30px;
  display: flex;
  justify-content: center;
  gap: 12px;
}

.time-slot-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 12px;
  max-width: 600px;
  margin: 0 auto;
}

.time-slot {
  padding: 16px;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  text-align: center;
}

.time-slot:hover:not(.disabled) {
  border-color: #409eff;
}

.time-slot.selected {
  border-color: #409eff;
  background-color: #ecf5ff;
}

.time-slot.disabled {
  background-color: #f5f5f5;
  cursor: not-allowed;
  opacity: 0.6;
}

.slot-time {
  display: block;
  font-size: 16px;
  font-weight: bold;
  color: #333;
  margin-bottom: 4px;
}

.slot-remaining {
  display: block;
  font-size: 12px;
  color: #999;
}

.slot-remaining.low-stock {
  color: #e6a23c;
}

.confirm-info {
  max-width: 500px;
  margin: 0 auto;
}
</style>
