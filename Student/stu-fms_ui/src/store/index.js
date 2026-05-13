import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

export default new Vuex.Store({
  state: sessionStorage.getItem('state') ? JSON.parse(sessionStorage.getItem('state')):{
    cache: {},
    showFramework: false, // 是否需要显示框架
    user: null
  },
  mutations: {
    // 把一个数据添加到缓存中
    // expirationTime:数据到期时间
    appendToCache (state, params) {
      if (params.timeout) {
        this.state.cache[params.key] = {
          data: params.data,
          expirationTime: new Date().getTime() + params.timeout
        }
      } else {
        this.state.cache[params.key] = { data: params.data }
      }
    },
    // 从缓存中删除一个数据
    removeFromCache (state, key) {
      delete this.state.cache.key
    },
    showFramework (state, isShow) {
      this.state.showFramework = isShow
    },
    setUser (state, user) {
      state.user = user
    }
  },
  actions: {
	  appendToCache (context, params) {
      context.commit('appendToCache', params)
	  }
  },
  modules: {
  },
  getters: {
    hasPermission(state,permission){
      console.log("权限:",permission,state);
    },
    getUserInfo (state) {
      return state.user
    },
	  // 获取缓存中的一个数据
	  getCacheData: (state) => (key) => {
		  var needRequest = true
		  var data = null
		  // 判断是否含有这个键
		  if (state.cache.hasOwnProperty(key)) {
			  var value = state.cache[key]
			 // 判断是否需要重新请求
			 if (value.hasOwnProperty('expirationTime')) {
				 needRequest = new Date().getTime() > value.expirationTime
			 } else {
				 needRequest = false
			 }
			 data = value.data
		  }
		  // needRequest用来记录是否需要重新发起请求来获取
		  return { data, needRequest }
	  }
  }
})
