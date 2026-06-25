import axios from 'axios'
import { ElMessage } from 'element-plus'
import router from '@/router'

const request = axios.create({
  timeout: 10000
})

request.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

request.interceptors.response.use(
  (response) => {
    // 后端返回 { code: 200, data: ..., message: ... }
    const res = response.data
    // 如果后端返回的 code 不是 200，也视为错误
    if (res && res.code !== 200) {
      ElMessage.error(res.message || '请求失败')
      return Promise.reject(new Error(res.message || '请求失败'))
    }
    return res
  },
  (error) => {
    if (error.response) {
      const { status, data } = error.response
      console.error(`[API错误] ${error.config?.method?.toUpperCase()} ${error.config?.url} → HTTP ${status}`, data)
      if (status === 401) {
        localStorage.removeItem('token')
        localStorage.removeItem('userInfo')
        router.push('/login')
        ElMessage.error('登录已过期，请重新登录')
      } else {
        // 显示后端返回的具体错误信息，或显示状态码
        const msg = data?.message || data?.msg || `请求失败(${status})`
        ElMessage.error(msg)
      }
    } else if (error.request) {
      // 请求已发出但没有响应（后端未启动/网络不通）
      console.error('[网络错误] 后端服务可能未启动:', error.config?.url)
      ElMessage.error('无法连接到后端服务，请确认后端已启动')
    } else {
      console.error('[请求配置错误]', error.message)
      ElMessage.error('请求配置错误')
    }
    return Promise.reject(error)
  }
)

export default request
