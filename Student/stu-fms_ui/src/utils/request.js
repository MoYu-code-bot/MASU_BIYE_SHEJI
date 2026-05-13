/*
封装所有的ajax请求
形成同一个错误响应,
同时实现统一的缓存效果
*/
import axios from 'axios'
import { Notification } from 'element-ui'

/**
 * 暴露出去的接口
 */
export default { ajax, get, post }

/**
 * 发起请求,返回数据
 * 这里不能使用缓存
 * @param {Object} params
 */
function json (params) {
  return axios(params)
    .then(res=>{
      if(res.data.status==-1){
         Notification({
          type: 'error',
          message: res.data.msg
        })
      }
      return res;
    })
    .catch(err => {
      Notification({
        type: 'error',
        message: '数据请求异常'
      })
      // 返回到下一级,便于之后的catch获取这个err
      return err
    })
}

/**
 * 利用vuex使用缓存
 * @param {Object} params
 * @param {Object} store
 */
function ajax (params) {
  // 如果不需要使用缓存就直接返回一个异步请求对象
  if (!params.hasOwnProperty('cache') || !params.cache) {
    return json(params)
      .then(res => {
        res.useCache = false
        return res
      })
  }
  // 使用缓存
  // 从缓存里面拿到这个键对应的值
  /* cacheRes:{
		data:data //缓存的数据
		needRequest:true//需不需要发送请求
	}
	*/
  var store = params.cache.store
  var cacheRes = store.getters.getCacheData(params.cache.key)
  // 判断需不需要去发起请求
  if (cacheRes.needRequest == true) {
    return json(params)
      .then(res => {
        store.dispatch('appendToCache', { data: res, ...params.cache })
        res.useCache = false
        return res
      })
  } else {
    // 使用promise便于返回一个统一的链式调用接口
    return new Promise(function (resolve, reject) {
      resolve({ ...cacheRes.data, useCache: true })
    })
  }
}

/**
 * 封装好的get
 * @param {Object} url
 * @param {Object} data
 * @param {Object} cache
 */
function get (url, data, cache) {
  return ajax({
    url,
    params:(data ? data.params:{}),
    cache,
    method: 'get'
  })
}

/**
 * 封装好的post
 * @param {Object} url
 * @param {Object} data
 * @param {Object} cache
 */
function post (url, data, cache) {
  return ajax({
    url,
    data,
    cache,
    method: 'post'
  })
}
