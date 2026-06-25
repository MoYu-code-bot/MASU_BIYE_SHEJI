<template>
  <div class="login-container">
    <div class="login-box">
      <h1 class="login-title">火锅到家</h1>
      <el-tabs v-model="activeTab" class="login-tabs">
        <!-- 管理员登录 -->
        <el-tab-pane label="管理员登录" name="admin">
          <el-form :model="adminForm" :rules="adminRules" ref="adminFormRef" @keyup.enter="handleAdminLogin">
            <el-form-item prop="username">
              <el-input v-model="adminForm.username" placeholder="请输入用户名" prefix-icon="User" size="large" />
            </el-form-item>
            <el-form-item prop="password">
              <el-input v-model="adminForm.password" type="password" placeholder="请输入密码" prefix-icon="Lock" size="large" show-password />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" size="large" style="width: 100%;" @click="handleAdminLogin" :loading="loading">
                登录
              </el-button>
            </el-form-item>
          </el-form>
        </el-tab-pane>

        <!-- 顾客登录 -->
        <el-tab-pane label="顾客登录" name="customer">
          <el-form :model="customerForm" :rules="customerRules" ref="customerFormRef" @keyup.enter="handleCustomerLogin">
            <el-form-item prop="phone">
              <el-input v-model="customerForm.phone" placeholder="请输入手机号" prefix-icon="Phone" size="large" />
            </el-form-item>
            <el-form-item prop="password">
              <el-input v-model="customerForm.password" type="password" placeholder="请输入密码" prefix-icon="Lock" size="large" show-password />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" size="large" style="width: 100%;" @click="handleCustomerLogin" :loading="loading">
                登录
              </el-button>
            </el-form-item>
          </el-form>
          <div class="register-link">
            <router-link to="/register">没有账号？立即注册</router-link>
          </div>
        </el-tab-pane>
      </el-tabs>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { adminLogin, customerLogin as customerLoginApi } from '@/api'

const router = useRouter()
const activeTab = ref('customer')
const loading = ref(false)

const adminFormRef = ref(null)
const customerFormRef = ref(null)

const adminForm = ref({
  username: '',
  password: ''
})

const customerForm = ref({
  phone: '',
  password: ''
})

const adminRules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }]
}

const customerRules = {
  phone: [
    { required: true, message: '请输入手机号', trigger: 'blur' },
    { pattern: /^1[3-9]\d{9}$/, message: '手机号格式不正确', trigger: 'blur' }
  ],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }]
}

const handleAdminLogin = async () => {
  await adminFormRef.value.validate()
  loading.value = true
  try {
    const res = await adminLogin(adminForm.value)
    const data = res.data || {}
    const token = data.token || res.token
    const role = data.role || 'ADMIN'    // 从后端响应读取角色，不再硬编码
    localStorage.setItem('token', token)
    localStorage.setItem('role', role)
    localStorage.setItem('username', adminForm.value.username)
    // 存储完整用户信息（含storeId等）
    localStorage.setItem('userInfo', JSON.stringify({
      id: data.id,
      username: data.username || adminForm.value.username,
      realName: data.realName,
      role: role,
      storeId: data.storeId,
      avatar: data.avatar
    }))
    ElMessage.success('登录成功')
    router.push('/admin/dashboard')
  } catch (e) {
    ElMessage.error('登录失败，请检查用户名和密码')
  } finally {
    loading.value = false
  }
}

const handleCustomerLogin = async () => {
  await customerFormRef.value.validate()
  loading.value = true
  try {
    const res = await customerLoginApi(customerForm.value)
    const token = typeof res.data === 'string' ? res.data : (res.data?.token || res.token)
    localStorage.setItem('token', token)
    localStorage.setItem('role', 'CUSTOMER')
    localStorage.setItem('customerPhone', customerForm.value.phone)
    ElMessage.success('登录成功')
    router.push('/c')
  } catch (e) {
    ElMessage.error('登录失败，请检查手机号和密码')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-container {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.login-box {
  width: 420px;
  background: #fff;
  border-radius: 12px;
  padding: 40px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
}

.login-title {
  text-align: center;
  margin-bottom: 30px;
  font-size: 28px;
  color: #333;
}

.login-tabs :deep(.el-tabs__nav) {
  width: 100%;
  display: flex;
}

.login-tabs :deep(.el-tabs__item) {
  flex: 1;
  text-align: center;
}

.register-link {
  text-align: center;
  margin-top: 12px;
}

.register-link a {
  color: #409eff;
  text-decoration: none;
  font-size: 14px;
}

.register-link a:hover {
  text-decoration: underline;
}
</style>
