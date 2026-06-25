<template>
  <el-container class="admin-layout">
    <!-- 侧边栏 -->
    <el-aside width="220px" class="admin-aside">
      <div class="logo-area">
        <h2>火锅到家管理系统</h2>
      </div>
      <el-menu
        :default-active="activeMenu"
        router
        background-color="#304156"
        text-color="#bfcbd9"
        active-text-color="#409eff"
        class="admin-menu"
      >
        <el-menu-item index="/admin/dashboard">
          <el-icon><DataLine /></el-icon>
          <span>数据看板</span>
        </el-menu-item>
        <el-menu-item v-if="hasMenuAccess(['ADMIN','MANAGER'])" index="/admin/stores">
          <el-icon><OfficeBuilding /></el-icon>
          <span>门店管理</span>
        </el-menu-item>
        <el-menu-item v-if="hasMenuAccess(['ADMIN','MANAGER'])" index="/admin/categories">
          <el-icon><Menu /></el-icon>
          <span>分类管理</span>
        </el-menu-item>
        <el-menu-item v-if="hasMenuAccess(['ADMIN','MANAGER'])" index="/admin/dishes">
          <el-icon><Food /></el-icon>
          <span>菜品管理</span>
        </el-menu-item>
        <el-menu-item index="/admin/reservations">
          <el-icon><Calendar /></el-icon>
          <span>预订管理</span>
        </el-menu-item>
        <el-menu-item index="/admin/reviews">
          <el-icon><ChatDotRound /></el-icon>
          <span>评价管理</span>
        </el-menu-item>
        <el-menu-item v-if="hasMenuAccess(['ADMIN','MANAGER'])" index="/admin/banners">
          <el-icon><PictureFilled /></el-icon>
          <span>轮播图管理</span>
        </el-menu-item>
        <el-menu-item v-if="hasMenuAccess(['ADMIN','MANAGER'])" index="/admin/announcements">
          <el-icon><Bell /></el-icon>
          <span>公告管理</span>
        </el-menu-item>
        <el-menu-item v-if="hasMenuAccess(['ADMIN','MANAGER'])" index="/admin/sysusers">
          <el-icon><User /></el-icon>
          <span>系统用户管理</span>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <!-- 右侧主区域 -->
    <el-container>
      <el-header class="admin-header">
        <div class="header-left">
          <span class="page-title">火锅到家管理系统</span>
        </div>
        <div class="header-right">
          <el-tag :type="roleTagType" size="small" class="role-tag">{{ roleLabel }}</el-tag>
          <span class="welcome-text">欢迎，{{ username }}</span>
          <el-button type="danger" size="small" @click="handleLogout">
            <el-icon><SwitchButton /></el-icon>
            退出登录
          </el-button>
        </div>
      </el-header>
      <el-main class="admin-main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  DataLine,
  OfficeBuilding,
  Menu,
  Food,
  Calendar,
  ChatDotRound,
  PictureFilled,
  Bell,
  User,
  SwitchButton
} from '@element-plus/icons-vue'

const router = useRouter()
const route = useRoute()

const activeMenu = computed(() => route.path)

const currentRole = computed(() => {
  return localStorage.getItem('role') || 'STAFF'
})

const username = computed(() => {
  return localStorage.getItem('username') || '用户'
})

const roleLabel = computed(() => {
  const map = { ADMIN: '系统管理员', MANAGER: '店长', STAFF: '员工' }
  return map[currentRole.value] || currentRole.value
})

const roleTagType = computed(() => {
  const map = { ADMIN: 'danger', MANAGER: 'warning', STAFF: 'info' }
  return map[currentRole.value] || 'info'
})

const hasMenuAccess = (allowedRoles) => {
  return allowedRoles.includes(currentRole.value)
}

const handleLogout = () => {
  ElMessageBox.confirm('确定要退出登录吗？', '提示', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'warning'
  }).then(() => {
    localStorage.removeItem('token')
    localStorage.removeItem('username')
    localStorage.removeItem('role')
    localStorage.removeItem('userInfo')
    localStorage.removeItem('customerPhone')
    ElMessage.success('已退出登录')
    router.push('/login')
  }).catch(() => {})
}
</script>

<style scoped>
.admin-layout {
  height: 100vh;
}

.admin-aside {
  background-color: #304156;
  overflow-y: auto;
}

.logo-area {
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #263445;
}

.logo-area h2 {
  color: #fff;
  font-size: 16px;
  margin: 0;
  white-space: nowrap;
}

.admin-menu {
  border-right: none;
}

.admin-header {
  background-color: #fff;
  box-shadow: 0 1px 4px rgba(0, 21, 41, 0.08);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
}

.page-title {
  font-size: 18px;
  font-weight: bold;
  color: #333;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.role-tag {
  font-weight: bold;
}

.welcome-text {
  font-size: 14px;
  color: #666;
}

.admin-main {
  background-color: #f0f2f5;
  padding: 20px;
}
</style>
