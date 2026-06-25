import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  // 根路径重定向到管理后台
  {
    path: '/',
    redirect: '/admin/dashboard'
  },
  // 后台管理路由（需要登录，按角色区分权限）
  {
    path: '/admin',
    component: () => import('@/views/admin/Layout.vue'),
    redirect: '/admin/dashboard',
    meta: { requiresAuth: true },
    children: [
      { path: 'dashboard', name: 'Dashboard', component: () => import('@/views/admin/DashboardView.vue'), meta: { title: '数据看板', roles: ['ADMIN', 'MANAGER', 'STAFF'] } },
      { path: 'stores', name: 'StoreManage', component: () => import('@/views/admin/StoreManage.vue'), meta: { title: '门店管理', roles: ['ADMIN', 'MANAGER'] } },
      { path: 'categories', name: 'CategoryManage', component: () => import('@/views/admin/CategoryManage.vue'), meta: { title: '分类管理', roles: ['ADMIN', 'MANAGER'] } },
      { path: 'dishes', name: 'DishManage', component: () => import('@/views/admin/DishManage.vue'), meta: { title: '菜品管理', roles: ['ADMIN', 'MANAGER'] } },
      { path: 'reservations', name: 'ReservationManage', component: () => import('@/views/admin/ReservationManage.vue'), meta: { title: '预订管理', roles: ['ADMIN', 'MANAGER', 'STAFF'] } },
      { path: 'reviews', name: 'ReviewManage', component: () => import('@/views/admin/ReviewManage.vue'), meta: { title: '评价管理', roles: ['ADMIN', 'MANAGER', 'STAFF'] } },
      { path: 'banners', name: 'BannerManage', component: () => import('@/views/admin/BannerManage.vue'), meta: { title: '轮播图管理', roles: ['ADMIN', 'MANAGER'] } },
      { path: 'announcements', name: 'AnnouncementManage', component: () => import('@/views/admin/AnnouncementManage.vue'), meta: { title: '公告管理', roles: ['ADMIN', 'MANAGER'] } },
      { path: 'sysusers', name: 'SysUserManage', component: () => import('@/views/admin/SysUserManage.vue'), meta: { title: '系统用户管理', roles: ['ADMIN', 'MANAGER'] } }
    ]
  },
  // 登录/注册
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/auth/LoginView.vue')
  },
  {
    path: '/register',
    name: 'Register',
    component: () => import('@/views/auth/RegisterView.vue')
  },
  // 顾客端路由
  {
    path: '/c',
    component: () => import('@/views/customer/CustomerLayout.vue'),
    children: [
      { path: '', name: 'CustomerHome', component: () => import('@/views/customer/CustomerHome.vue'), meta: { title: '顾客首页' } },
      { path: 'stores/:storeId', name: 'StoreDetail', component: () => import('@/views/customer/StoreDetail.vue') },
      { path: 'reserve/:storeId', name: 'ReservePage', component: () => import('@/views/customer/ReservePage.vue'), meta: { requiresCustomerAuth: true } },
      { path: 'my-reservations', name: 'MyReservations', component: () => import('@/views/customer/MyReservations.vue'), meta: { requiresCustomerAuth: true } }
    ]
  },
  // 404 兜底
  {
    path: '/:pathMatch(.*)*',
    redirect: '/admin/dashboard'
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// 路由守卫：认证 + 角色权限
router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('token')
  const role = localStorage.getItem('role')

  // B端管理后台路由守卫
  if (to.path.startsWith('/admin') && to.meta.roles) {
    if (!token) {
      // 未登录 → 跳转登录页
      next({ path: '/login', query: { redirect: to.fullPath } })
      return
    }
    if (Array.isArray(to.meta.roles) && !to.meta.roles.includes(role)) {
      // 无权限 → 403提示
      next({ path: '/admin/dashboard', query: {} })
      return
    }
  }

  // 顾客端需要认证的路由
  if (to.meta.requiresCustomerAuth && (!token || role !== 'CUSTOMER')) {
    next({ path: '/login', query: { redirect: to.fullPath } })
    return
  }

  // 已登录管理员访问登录页 → 重定向到后台
  if (to.path === '/login' && token && role !== 'CUSTOMER') {
    next('/admin/dashboard')
    return
  }

  next()
})

export default router
