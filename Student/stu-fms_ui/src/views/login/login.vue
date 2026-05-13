<template>
	<div id="login">
		<!-- 文字部分 -->
		<div id="font">
			克勤之致<br>
			<span class="yellow">结青藤</span>之缘
		</div>
		<!-- 图片部分 -->
		<div id="img">
			<img src="../../assets/images/lh/login-sc.png">
		</div>
		<!-- 下部图片 -->
		<div id="downImg">
			<img src="../../assets/images/lh/login-cao.png">
		</div>
		<!-- 登录部分 -->
		<div id="account" v-loading="loginLoading">
			<h1 class="title">学生成长电子档案系统</h1>
			<el-form @submit.native.prevent label-width="80px" ref="ruleForm" :model="ruleForm" :rules="rules">
				<el-form-item label="账号" prop="account">
					<el-input autocomplete="on" @keyup.enter.native="submitForm('ruleForm')" placeholder="账号" prefix-icon="el-icon-user" v-model="ruleForm.account">
					</el-input>
				</el-form-item>
				<el-form-item label="密码" prop="password">
					<el-input autocomplete="on" @keyup.enter.native="submitForm('ruleForm')" placeholder="密码" prefix-icon="el-icon-warning-outline" v-model="ruleForm.password" show-password>
					</el-input>
				</el-form-item>
				<el-form-item>
					<el-button type="primary" @click="submitForm('ruleForm')">登录</el-button>
					<el-button native-type="reset" @click="resetForm('ruleForm')">重置</el-button>
				</el-form-item>
				<el-form-item label-width="0">
					<el-button type="text" @click="$router.push('/register')">没有账号？立即注册</el-button>
				</el-form-item>
			</el-form>
			<div class="lookout">
				<p>注意：</p>
				<p>1.学生登录请使用学号，初始密码为后6位。请登录之后及时修改。</p>
				<p>2.其他用户请使用管理员分配的账户进行登录。</p>
				<p>3.账号问题请联系:12452175@qq.com</p>
			</div>
		</div>
	</div>
</template>

<script>
  import  request from "@/utils/request"
  import { Notification } from 'element-ui'
export default {
  data () {
    /* 自定义验证 */
    var validateAccount = (rule, value, callback) => {
      var regAccount = /^[a-zA-Z0-9_]{4,15}$/
      if (value === '') {
        callback(new Error('账号不能为空!'))
      } else if (regAccount.test(value) == false) {
        callback(new Error('只允许4-15位字母数字下划线组合!'))
      } else {
        callback()
      }
    }
    var validatePassword = (rule, value, callback) => {
      if (value === '') {
        callback(new Error('密码不能为空!'))
      } else {
        callback()
      }
    }
    return {
      loginLoading:false,
      ruleForm: {
        account: '',
        password: ''
      },
      rules: {
        account: [{
          validator: validateAccount,
          trigger: 'blur'
        }],
        password: [{
          validator: validatePassword,
          trigger: 'blur'
        }]
      }
    }
  },
  methods: {
    submitForm (formName) {
      this.$refs[formName].validate((valid) => {
        if (valid) {
          // 开始登录
          this.loginLoading=true;
          request.post("/api/auth/login",{
            username:this.ruleForm.account,
            password:this.ruleForm.password
          }).then(res => {
            this.loginLoading=false;
            if(res.data.status==-1) return;
            Notification({
              type:"success",
              message:res.data.msg
            });
            // 同步添加用户信息
            this.$store.state.user=res.data.data
            // this.$store.commit('setUser', res.data.data)
            //初始化shiro
            var self=this;
            this.$shiro.initShiro(this,function(){
              //请求审核状态信息
              self.$dict.initDict(self,function(){
                //跳转到首页
                self.$router.push('/firstPage')
              })
            });
          });
        } else {
          return false
        }
      })
    },
    resetForm (formName) {
      this.$refs[formName].resetFields()
    }
  },
  mounted () {
  }
}
</script>

<style lang="less">
	/*声明动画*/
	@keyframes moveImg {
		0% {
			top: 0px
		}

		50% {
			top: 10px
		}

		100% {
			top: 0px
		}
	}

	#login {
		width: 100%;
		height: 100%;
		background: #10c55c;

		#font {
			position: relative;
			width: 500px;
			left: 160px;
			top: 170px;
			font-size: 5rem;
			color: white;

			.yellow {
				color: yellow;
			}
		}

		#img {
			position: relative;
			left: 35%;
			top: -30%;
			width: 400px;
			height: 350px;

			img {
				animation: moveImg 3s infinite;
				position: absolute;
				top: 0px;
				left: 0px;
				width: 400px;
				height: 350px;
			}
		}

		#downImg {
			position: absolute;
			bottom: 0px;
			left: 150px;
		}

		#account {
			position: absolute;
			display: inline-block;
			width: 400px;
			height: 460px;
			top: 80px;
			right: 12%;
			background: white;
			border-radius: 10px;
			box-shadow: 3px 3px 10px 2px #0b9243;
			padding: 10px;

			.title {
				color: #777;
				font-size: 35px;
				text-align: center;
				margin: 35px 0px;
			}

			.lookout {
				color: #777;
				padding: 20px;
			}

			.el-input {
				width: 85%;
			}

		}
	}
</style>
