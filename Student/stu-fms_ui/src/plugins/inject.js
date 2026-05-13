/*注入的js*/
/*把这些都挂载到vue实例下*/
import shiro from "./shiro"
import dict from './dict'
import echarts from "echarts"

export default {
  install: (Vue, options) => {
    Vue.prototype.$shiro = shiro
    Vue.prototype.$dict = dict
    Vue.prototype.$echarts = echarts
  }
}
