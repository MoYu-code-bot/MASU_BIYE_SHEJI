<template>
	<div id="framework">
		<!-- 侧边栏 -->
		<div id="sidebar">
			<div id="logo" @click="toIndex">
				<!-- logo -->
				<el-avatar v-show="isCollapse" shape="square" :size="48" src="https://cn.vuejs.org/images/logo.png"></el-avatar>
				<span v-show="!isCollapse">档案管理系统</span>
			</div>
			<el-menu class="el-menu-vertical-demo" @select="handleSelect" :collapse="isCollapse" :collapse-transition="true"
			 background-color="#20222a" text-color="#bdbdc0">
        <el-menu-item  index="/firstPage">
          <i class="el-icon-s-home"></i>
          <span slot="title">首页</span>
        </el-menu-item>
        <el-menu-item v-if="$shiro.hasPermission(this,'menu:stu_info')" index="/basic">
					<i class="el-icon-user"></i>
					<span slot="title">基本信息</span>
				</el-menu-item>
				<el-submenu v-if="$shiro.hasPermission(this,'menu:course_project')" index="2">
					<template slot="title">
						<i class="el-icon-s-grid"></i>
						<span slot="title">课程学业信息</span>
					</template>
					<el-menu-item  index="/courseMes">
						<i class="el-icon-tickets"></i>
						<span slot="title">课业成绩</span>
					</el-menu-item>
					<el-menu-item index="/scienceMes">
						<i class="el-icon-tickets"></i>
						<span slot="title">科研项目</span>
					</el-menu-item>
				</el-submenu>
				<el-menu-item v-if="$shiro.hasPermission(this,'menu:activity')" index="/activity">
					<i class="el-icon-s-flag"></i>
					<span slot="title">活动/竞赛信息</span>
				</el-menu-item>
				<el-menu-item v-if="$shiro.hasPermission(this,'menu:medical')"  index="/medical">
					<i class="el-icon-scissors"></i>
					<span slot="title">健康体检信息</span>
				</el-menu-item>
				<el-menu-item v-if="$shiro.hasPermission(this,'menu:weijiMes')" index="/weijiMes">
					<i class="el-icon-info"></i>
					<span slot="title">违纪信息</span>
				</el-menu-item>
				<el-menu-item v-if="$shiro.hasPermission(this,'menu:grow_infro')" index="/grow_infro">
					<i class="el-icon-star-on"></i>
					<span slot="title">就业与发展信息</span>
				</el-menu-item>
				<el-submenu v-if="$shiro.hasPermission(this,'menu:archives')" index="10">
          <template slot="title">
            <i class="el-icon-s-management"></i>
            <span slot="title">档案管理</span>
          </template>
          <el-submenu v-if="$shiro.hasPermission(this,'menu:archives')" index="11">
            <template slot="title">
              <i class="el-icon-s-finance"></i>
              <span slot="title">档案分析</span>
            </template>
            <el-menu-item index="/forecast">
              <i class="el-icon-s-marketing"></i>
              <span slot="title">就业趋势预测</span>
            </el-menu-item>
            <el-menu-item index="/analyze">
              <i class="el-icon-s-marketing"></i>
              <span slot="title">指标分析</span>
            </el-menu-item>
          </el-submenu>
          <el-menu-item index="/archives-manager">
            <i class="el-icon-s-management"></i>
            <span slot="title">档案借阅管理</span>
          </el-menu-item>
          <el-menu-item  index="/archives">
            <i class="el-icon-s-finance"></i>
            <span slot="title">档案管理</span>
          </el-menu-item>
				</el-submenu>
        <el-menu-item v-if="$shiro.hasPermission(this,'menu:archives_borrow')"  index="/borrow">
          <i class="el-icon-s-finance"></i>
          <span slot="title">档案借阅</span>
        </el-menu-item>
				<el-submenu v-if="$shiro.hasPermission(this,'menu:system_manager')" index="8">
					<template  slot="title">
						<i class="el-icon-location"></i>
						<span slot="title">系统管理</span>
					</template>
					<el-menu-item v-if="$shiro.hasPermission(this,'menu:user_manager')" index="/userManager">
						<i class="el-icon-document"></i>
						<span slot="title">用户管理</span>
					</el-menu-item>
					<el-menu-item v-if="$shiro.hasPermission(this,'menu:role_manager')" index="/roleManager">
						<i class="el-icon-document"></i>
						<span slot="title">角色管理</span>
					</el-menu-item>
					<el-menu-item v-if="$shiro.hasPermission(this,'menu:permission_manager')" index="/permission">
						<i class="el-icon-document"></i>
						<span slot="title">权限管理</span>
					</el-menu-item>
<!--					<el-menu-item v-if="$shiro.hasPermission(this,'menu:setting')" index="/setting">-->
<!--						<i class="el-icon-document"></i>-->
<!--						<span slot="title">系统设置</span>-->
<!--					</el-menu-item>-->
				</el-submenu>
			</el-menu>
		</div>
		<div id="right">
			<!-- 顶部栏 -->
			<div id="topbar">
				<el-menu default-active="2-1" class="el-menu-demo" mode="horizontal" @select="handleSelect" menu-trigger="hover">
					<el-menu-item index="1" @click="handleCollapse">
						<span v-show="isCollapse" class="el-icon-s-unfold"></span>
						<span v-show="!isCollapse" class="el-icon-s-fold"></span>
					</el-menu-item>
					<el-menu-item index="2"><span class="el-icon-school"></span>学校官网</el-menu-item>
					<el-menu-item index="3"><span class="el-icon-document-copy"></span>图书馆</el-menu-item>
					<div style="float: right;width:100px;height:60px;">
						<el-submenu index="4">
							<template slot="title">
								<el-avatar shape="square" :size="32" :src="avatar"></el-avatar>
							</template>
							<el-menu-item index="4-1"><span class="el-icon-user-solid"></span>基本信息</el-menu-item>
							<el-menu-item index="4-2"><span class="el-icon-key"></span>
								用户信息
							</el-menu-item>
							<el-divider></el-divider>
							<el-menu-item index="/logout"><span class="el-icon-back"></span>
								退出
							</el-menu-item>
						</el-submenu>
					</div>
				</el-menu>
			</div>
			<div id="mainOut">
				<!-- 主要内容区 -->
				<div id="main" class="panel">
					<slot></slot>
					<el-backtop target="#mainOut">
						<span class="backTop el-icon-caret-top"></span>
					</el-backtop>
				</div>
				<!--页脚-->
				<div id="footer" class="panel"></div>
			</div>
		</div>
	</div>
</template>

<script>
export default {
  name: 'framework',
  data () {
    return {
      isCollapse: true
    }
  },
  mounted(){
  },
  computed:{
    avatar(){
      try {
        if(this.$store.state.user){
          return this.$store.state.user.avatar;
        }
      }catch (e) {
      }
      return "http://localhost:8080/images/logo.png";
    }
  },
  methods: {
    handleCollapse () {
      // 切换显示与展示
      this.isCollapse = !this.isCollapse
    },
    handleSelect (key, keyPath) {
      var path = keyPath[keyPath.length - 1]
      // 退出登录
      if (path === '/logout') {
        this.$store.commit('setUser', null)
        this.$router.push('/login')
      } else {
        // 跳转到指定页面
        this.$router.push(path)
      }
    },
    toIndex () {
      this.$router.push('/')
    }

  },
  mounted () {
  }
}
</script>

<style lang="less" scoped>
	.el-avatar {
		background-color: rgba(0, 0, 0, 0);
	}

	#framework {
		width: 100%;
		height: 100%;
		display: flex;

		#sidebar {
			height: 100%;
			background: #20222a;
			overflow: auto;

			#logo {
				color: #bdbdc0;
				height: 60px;

				.el-menu {
					border: 0px;
				}

				span {
					background-color: rgba(0, 0, 0, 0);
					display: block;
					text-align: center;
					position: relative;
					top: 50%;
					left: 50%;
					transform: translate(-50%, -50%);
				}
			}
		}

		#right {
			flex: 1;
			background: #f2f2f2;
			overflow: hidden;

			#topbar {
				position: sticky;
				top: 0px;
				width: 100%;
			}

			#mainOut {
				height: calc(~"100% - 80px");
				;
				padding: 10px;
				padding-bottom: 0px;
				overflow-y: auto;

				.panel {
					background: white;
					border-radius: 5px;
				}

				#main {

					// height: 1500px;
					.backTop {
						font-size: 32px;
						color: #409eff;
					}
				}

				#footer {
					margin-top: 10px;
				}
			}
		}
	}
</style>
