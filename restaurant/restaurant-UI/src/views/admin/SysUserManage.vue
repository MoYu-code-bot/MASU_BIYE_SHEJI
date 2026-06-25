<template>
  <div class="sysuser-manage">
    <div class="page-header">
      <h2>系统用户管理</h2>
      <el-button type="primary" @click="handleAdd">
        <el-icon><Plus /></el-icon>
        新增用户
      </el-button>
    </div>

    <el-table :data="tableData" border stripe v-loading="loading" style="width: 100%">
      <el-table-column prop="id" label="ID" width="70" />
      <el-table-column prop="username" label="用户名" min-width="120" />
      <el-table-column prop="realName" label="姓名" min-width="100" />
      <el-table-column prop="phone" label="手机" width="130" />
      <el-table-column prop="role" label="角色" width="100">
        <template #default="{ row }">
          <el-tag :type="roleTagType(row.role)" size="small">
            {{ roleLabel(row.role) }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="status" label="状态" width="100">
        <template #default="{ row }">
          <el-switch
            :model-value="row.status === 1"
            :disabled="!canEdit(row)"
            @change="(val) => handleToggleStatus(row, val)"
            active-text="启用"
            inactive-text="禁用"
          />
        </template>
      </el-table-column>
      <el-table-column label="操作" width="180" fixed="right">
        <template #default="{ row }">
          <el-button size="small" type="primary" :disabled="!canEdit(row)" @click="handleEdit(row)">编辑</el-button>
          <el-popconfirm
            title="确定要删除该用户吗？"
            confirm-button-text="确定"
            cancel-button-text="取消"
            @confirm="handleDelete(row)"
            v-if="canDelete(row)"
          >
            <template #reference>
              <el-button size="small" type="danger">删除</el-button>
            </template>
          </el-popconfirm>
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
    <el-dialog v-model="dialogVisible" :title="isEdit ? '编辑用户' : '新增用户'" width="480px">
      <el-form :model="form" :rules="rules" ref="formRef" label-width="80px">
        <el-form-item label="用户名" prop="username">
          <el-input v-model="form.username" :disabled="isEdit" placeholder="请输入用户名" />
        </el-form-item>
        <el-form-item v-if="!isEdit" label="密码" prop="password">
          <el-input v-model="form.password" type="password" placeholder="请输入密码" show-password />
        </el-form-item>
        <el-form-item label="姓名" prop="realName">
          <el-input v-model="form.realName" placeholder="请输入姓名" />
        </el-form-item>
        <el-form-item label="手机" prop="phone">
          <el-input v-model="form.phone" placeholder="请输入手机号" />
        </el-form-item>
        <el-form-item label="角色" prop="role">
          <el-select v-model="form.role" style="width: 100%;">
            <el-option v-if="isAdmin" label="系统管理员" value="ADMIN" />
            <el-option v-if="isAdmin" label="店长" value="MANAGER" />
            <el-option label="员工" value="STAFF" />
          </el-select>
        </el-form-item>
        <el-form-item label="门店" prop="storeId" v-if="form.role === 'MANAGER' || form.role === 'STAFF'">
          <el-input-number v-model="form.storeId" :min="1" placeholder="门店ID" style="width: 100%;" />
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
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'
import { getSysUsers, createSysUser, updateSysUserInfo, deleteSysUser, updateSysUser } from '@/api'

const loading = ref(false)
const tableData = ref([])
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)

const dialogVisible = ref(false)
const submitting = ref(false)
const formRef = ref(null)
const isEdit = ref(false)

const currentRole = ref(localStorage.getItem('role') || 'STAFF')
const isAdmin = computed(() => currentRole.value === 'ADMIN')
const isManager = computed(() => currentRole.value === 'MANAGER')

const form = ref({
  id: null,
  username: '',
  password: '',
  realName: '',
  phone: '',
  role: 'STAFF',
  storeId: null
})

const rules = computed(() => ({
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: isEdit.value ? [] : [{ required: true, message: '请输入密码', trigger: 'blur' }, { min: 6, message: '密码不少于6位', trigger: 'blur' }],
  realName: [{ required: true, message: '请输入姓名', trigger: 'blur' }],
  phone: [{ required: true, message: '请输入手机号', trigger: 'blur' }, { pattern: /^1[3-9]\d{9}$/, message: '手机号格式不正确', trigger: 'blur' }],
  role: [{ required: true, message: '请选择角色', trigger: 'change' }]
}))

const roleLabel = (role) => {
  const map = { ADMIN: '系统管理员', MANAGER: '店长', STAFF: '员工' }
  return map[role] || role
}

const roleTagType = (role) => {
  const map = { ADMIN: 'danger', MANAGER: 'warning', STAFF: 'info' }
  return map[role] || 'info'
}

// 店长不能编辑/禁用ADMIN用户
const canEdit = (row) => {
  if (isManager.value && row.role === 'ADMIN') return false
  return true
}

// 只有ADMIN可以删除用户
const canDelete = (row) => {
  if (!isAdmin.value) return false
  if (row.role === 'ADMIN') return false
  return true
}

const loadData = async () => {
  loading.value = true
  try {
    const res = await getSysUsers({ page: currentPage.value, pageSize: pageSize.value })
    tableData.value = res.data?.records || res.data?.list || res.data || []
    total.value = res.data?.total || 0
  } catch (e) {
    ElMessage.error('加载用户列表失败')
  } finally {
    loading.value = false
  }
}

const handleAdd = () => {
  isEdit.value = false
  form.value = { id: null, username: '', password: '', realName: '', phone: '', role: 'STAFF', storeId: null }
  dialogVisible.value = true
}

const handleEdit = (row) => {
  isEdit.value = true
  form.value = { ...row, password: '' }
  dialogVisible.value = true
}

const handleSubmit = async () => {
  await formRef.value.validate()
  submitting.value = true
  try {
    if (isEdit.value) {
      // 编辑时只传可修改的字段
      await updateSysUserInfo(form.value.id, {
        realName: form.value.realName,
        phone: form.value.phone,
        role: form.value.role,
        storeId: form.value.storeId,
        status: form.value.status
      })
      ElMessage.success('更新成功')
    } else {
      await createSysUser(form.value)
      ElMessage.success('创建成功')
    }
    dialogVisible.value = false
    loadData()
  } catch (e) {
    const msg = e.response?.data?.message || (isEdit.value ? '更新失败' : '创建失败')
    ElMessage.error(msg)
  } finally {
    submitting.value = false
  }
}

const handleToggleStatus = async (row, val) => {
  try {
    await updateSysUser({ id: row.id, status: val ? 1 : 0 })
    ElMessage.success(val ? '已启用' : '已禁用')
    loadData()
  } catch (e) {
    const msg = e.response?.data?.message || '操作失败'
    ElMessage.error(msg)
  }
}

const handleDelete = async (row) => {
  try {
    await deleteSysUser(row.id)
    ElMessage.success('删除成功')
    loadData()
  } catch (e) {
    const msg = e.response?.data?.message || '删除失败'
    ElMessage.error(msg)
  }
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.sysuser-manage {
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
