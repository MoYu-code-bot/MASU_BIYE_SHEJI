import Vue from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
import inject from './plugins/inject'

// 导入完整的element-ui
Vue.use(ElementUI)
//挂载到vue上
Vue.use(inject)

Vue.config.productionTip = true
window.vue=new Vue({
  router,
  store,
  render: h => h(App)
}).$mount('#app')

// 全局路由守卫
router.beforeEach((to, form, next) => {
  const isAuthFree = to.path === '/login' || to.path === '/' || to.path === '/register'
  // 已登录或登录/注册等公开页
  if (router.app.$store.state.user || isAuthFree) {
    if (to.path === '/login' || to.path === '/' || to.path === '/register') {
      router.app.$store.commit('showFramework', false)
    } else {
      router.app.$store.commit('showFramework', true)
    }
    next()
  } else {
    router.app.$store.commit('showFramework', false)
    //跳转到登录页面
    console.log(to)
    next({
      path:"/login"
    })
  }
})
