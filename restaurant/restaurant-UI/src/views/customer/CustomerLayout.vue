<template>
  <div class="customer-layout">
    <!-- 顶部导航 -->
    <header class="customer-header">
      <div class="header-content">
        <div class="logo" @click="$router.push('/c')">
          <span class="logo-text">火锅到家</span>
        </div>
        <nav class="header-nav">
          <router-link to="/c" class="nav-link">首页</router-link>
          <router-link to="/c/my-reservations" class="nav-link">我的预订</router-link>
        </nav>
        <div class="user-info">
          <template v-if="isLoggedIn">
            <span class="user-phone">{{ customerPhone }}</span>
            <el-button type="text" @click="handleLogout" size="small">退出</el-button>
          </template>
          <template v-else>
            <router-link to="/login" class="nav-link">登录</router-link>
          </template>
        </div>
      </div>
    </header>

    <!-- 主内容 -->
    <main class="customer-main">
      <router-view />
    </main>

    <!-- 底部 -->
    <footer class="customer-footer">
      <div class="footer-content">
        <p>火锅到家 - 美味火锅送到您身边</p>
        <p class="footer-copyright">Copyright 2024</p>
      </div>
    </footer>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'

const router = useRouter()

const isLoggedIn = computed(() => {
  return !!localStorage.getItem('token')
})

const customerPhone = computed(() => {
  return localStorage.getItem('customerPhone') || ''
})

const handleLogout = () => {
  localStorage.removeItem('token')
  localStorage.removeItem('role')
  localStorage.removeItem('customerPhone')
  ElMessage.success('已退出登录')
  router.push('/login')
}
</script>

<style scoped>
.customer-layout {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background-color: #f5f5f5;
}

.customer-header {
  background: #fff;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  position: sticky;
  top: 0;
  z-index: 100;
}

.header-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.logo {
  cursor: pointer;
}

.logo-text {
  font-size: 22px;
  font-weight: bold;
  color: #ff6b35;
}

.header-nav {
  display: flex;
  gap: 24px;
}

.nav-link {
  color: #333;
  text-decoration: none;
  font-size: 15px;
  transition: color 0.2s;
}

.nav-link:hover,
.nav-link.router-link-active {
  color: #ff6b35;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.user-phone {
  font-size: 14px;
  color: #666;
}

.customer-main {
  flex: 1;
  max-width: 1200px;
  width: 100%;
  margin: 0 auto;
  padding: 20px;
}

.customer-footer {
  background: #333;
  color: #fff;
  padding: 30px 20px;
  text-align: center;
}

.footer-content p {
  margin: 5px 0;
}

.footer-copyright {
  color: #999;
  font-size: 13px;
}
</style>
