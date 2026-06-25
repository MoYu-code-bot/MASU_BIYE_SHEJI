<template>
  <div class="store-manage">
    <div class="page-header">
      <h2>门店管理</h2>
      <el-button type="primary" @click="handleAdd">
        <el-icon><Plus /></el-icon>
        新增门店
      </el-button>
    </div>

    <el-table :data="tableData" border stripe v-loading="loading" style="width: 100%">
      <el-table-column prop="id" label="ID" width="70" />
      <el-table-column prop="name" label="名称" min-width="120" />
      <el-table-column prop="address" label="地址" min-width="200" />
      <el-table-column prop="phone" label="电话" width="130" />
      <el-table-column prop="businessHours" label="营业时间" width="150" />
      <el-table-column prop="avgPrice" label="人均消费" width="100">
        <template #default="{ row }">
          ¥{{ row.avgPrice }}
        </template>
      </el-table-column>
      <el-table-column prop="status" label="状态" width="100">
        <template #default="{ row }">
          <el-tag :type="row.status === 1 ? 'success' : 'danger'">
            {{ row.status === 1 ? '营业中' : '已停业' }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="220" fixed="right">
        <template #default="{ row }">
          <el-button size="small" type="primary" @click="handleEdit(row)">编辑</el-button>
          <el-button
            size="small"
            :type="row.status === 1 ? 'warning' : 'success'"
            @click="handleToggleStatus(row)"
          >
            {{ row.status === 1 ? '停业' : '营业' }}
          </el-button>
          <el-button size="small" type="danger" @click="handleDelete(row)">删除</el-button>
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

    <!-- 新增/编辑对话框 -->
    <el-dialog v-model="dialogVisible" :title="isEdit ? '编辑门店' : '新增门店'" width="500px">
      <el-form :model="form" :rules="rules" ref="formRef" label-width="100px">
        <el-form-item label="门店名称" prop="name">
          <el-input v-model="form.name" placeholder="请输入门店名称" />
        </el-form-item>
        <el-form-item label="地址" prop="address">
          <el-input v-model="form.address" placeholder="请输入地址" />
        </el-form-item>
        <el-form-item label="电话" prop="phone">
          <el-input v-model="form.phone" placeholder="请输入电话" />
        </el-form-item>
        <el-form-item label="营业时间" prop="businessHours">
          <el-input v-model="form.businessHours" placeholder="如 10:00-22:00" />
        </el-form-item>
        <el-form-item label="人均消费" prop="avgPrice">
          <el-input-number v-model="form.avgPrice" :min="0" :precision="0" />
        </el-form-item>
        <el-form-item label="封面图片" prop="coverImage">
          <el-input v-model="form.coverImage" placeholder="请输入图片URL" />
        </el-form-item>
        <el-form-item label="状态" prop="status">
          <el-radio-group v-model="form.status">
            <el-radio :value="1">营业</el-radio>
            <el-radio :value="0">停业</el-radio>
          </el-radio-group>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="handleSubmit" :loading="submitting">确定</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'
import { adminGetStores, createStore, updateStore, deleteStore } from '@/api'

const loading = ref(false)
const tableData = ref([])
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)

const dialogVisible = ref(false)
const isEdit = ref(false)
const submitting = ref(false)
const formRef = ref(null)

const defaultForm = {
  name: '',
  address: '',
  phone: '',
  businessHours: '',
  avgPrice: 0,
  coverImage: '',
  status: 1
}

const form = ref({ ...defaultForm })

const rules = {
  name: [{ required: true, message: '请输入门店名称', trigger: 'blur' }],
  address: [{ required: true, message: '请输入地址', trigger: 'blur' }],
  phone: [{ required: true, message: '请输入电话', trigger: 'blur' }]
}

const loadData = async () => {
  loading.value = true
  try {
    const res = await adminGetStores({ pageNum: currentPage.value, pageSize: pageSize.value })
    tableData.value = res.data?.list || []
    total.value = res.data?.total || 0
  } catch (e) {
    ElMessage.error('加载门店列表失败')
  } finally {
    loading.value = false
  }
}

const handleAdd = () => {
  isEdit.value = false
  form.value = { ...defaultForm }
  dialogVisible.value = true
}

const handleEdit = (row) => {
  isEdit.value = true
  form.value = { ...row }
  dialogVisible.value = true
}

const handleSubmit = async () => {
  await formRef.value.validate()
  submitting.value = true
  try {
    if (isEdit.value) {
      await updateStore(form.value)
      ElMessage.success('更新成功')
    } else {
      await createStore(form.value)
      ElMessage.success('创建成功')
    }
    dialogVisible.value = false
    loadData()
  } catch (e) {
    ElMessage.error(isEdit.value ? '更新失败' : '创建失败')
  } finally {
    submitting.value = false
  }
}

const handleDelete = (row) => {
  ElMessageBox.confirm(`确定删除门店「${row.name}」吗？`, '提示', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'warning'
  }).then(async () => {
    try {
      await deleteStore(row.id)
      ElMessage.success('删除成功')
      loadData()
    } catch (e) {
      ElMessage.error('删除失败')
    }
  }).catch(() => {})
}

const handleToggleStatus = async (row) => {
  const newStatus = row.status === 1 ? 0 : 1
  try {
    await updateStore(row.id, { ...row, status: newStatus })
    ElMessage.success(newStatus === 1 ? '已设为营业' : '已设为停业')
    loadData()
  } catch (e) {
    ElMessage.error('状态更新失败')
  }
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.store-manage {
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
</style>
