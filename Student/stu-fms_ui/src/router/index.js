import Vue from 'vue'
import VueRouter from 'vue-router'

Vue.use(VueRouter)

const routes = [
  {
    path: '/firstPage',
    name: 'firstPage',
    component: () => import('@/views/firstPage.vue')
  },
  {
    path: '/analyze',
    name: 'analyze',
    component: () => import('@/views/analyze.vue')
  },
  {
    path: '/forecast',
    name: 'forecast',
    component: () => import('@/views/forecast.vue')
  },
  {
    path: '/permission',
    name: 'permission',
    component: () => import('@/views/system/permission.vue')
  },
  {
    path: '/borrow',
    name: 'borrow',
    component: () => import('@/views/borrow.vue')
  },
  {
    path: '/',
    name: 'login',
    component: () => import('@/views/login/login.vue')
  },
  {
	  path: '/login',
	  name: 'login',
	  component: () => import('@/views/login/login.vue')
  },
  {
    path: '/register',
    name: 'register',
    component: () => import('@/views/login/register.vue')
  },
  {
    path: '/index',
    name: 'index',
    component: () => import('@/views/studentMes/basic.vue')
  },
  {
    path: '/userManager',
    name: 'userManager',
    component: () => import('@/views/system/userManager.vue')
  },
  {
    path: '/roleManager',
    name: 'roleManager',
    component: () => import('@/views/system/roleManager.vue')
  },
  {
    path: '/basic',
    name: 'basic',
    component: () => import('@/views/studentMes/basic.vue')
  },
  {
    path: '/courseMes',
    name: 'courseMes',
    component: () => import('@/views/studentMes/courseMes.vue')
  },
  {
    path: '/scienceMes',
    name: 'scienceMes',
    component: () => import('@/views/studentMes/scienceMes.vue')
  },
  {
    path: '/activity',
    name: 'activity',
    component: () => import('@/views/studentMes/activity.vue')
  },
  {
    path: '/medical',
    name: 'medical',
    component: () => import('@/views/studentMes/medical.vue')
  },
  {
    path: '/weijiMes',
    name: 'weijiMes',
    component: () => import('@/views/studentMes/weijiMes.vue')
  },
  {
    path: '/grow_infro',
    name: 'grow_infro',
    component: () => import('@/views/studentMes/grow_infro.vue')
  }, {
    path: '/archives',
    name: 'archives',
    component: () => import('@/views/archives-manager.vue')
  },
  {
    path: '/archives-manager',
    name: 'archives-manager',
    component: () => import('@/views/archives-borrow.vue')

  }
]

const router = new VueRouter({
  mode: 'hash',//只有hash模式才能保证地址栏输入时会走路由守卫
  base: process.env.BASE_URL,
  routes
})

export default router
