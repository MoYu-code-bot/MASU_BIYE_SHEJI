<template>
	<div id="app">
		<template v-if="$store.state.showFramework">
			<framework >
				<!-- 主要内容区域 -->
				<Router-view></Router-view>
			</framework>
		</template>
		<template v-else>
			<Router-view ></Router-view>
		</template >
	</div>
</template>
<script>
import framework from '@/components/framework'
export default {
  components: { framework },
  mounted() {
    window.addEventListener('unload', this.unLoad)
    window.addEventListener('load', this.load)
  },
  methods: {
    load(){
      //刷新shiro
      this.$shiro.initShiro(this);
    },
    unLoad() {
      //保存store
      sessionStorage.setItem('state', JSON.stringify(this.$store.state))
    }
  }
}
</script>

<style lang="less">
</style>
