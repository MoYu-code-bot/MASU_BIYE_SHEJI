<template>
	<div id="login">
		<div id="font">
			克勤之致<br>
			<span class="yellow">结青藤</span>之缘
		</div>
		<div id="img">
			<img src="../../assets/images/lh/login-sc.png">
		</div>
		<div id="downImg">
			<img src="../../assets/images/lh/login-cao.png">
		</div>
		<div id="account" v-loading="registerLoading">
			<h1 class="title">用户注册</h1>
			<el-form @submit.native.prevent label-width="80px" ref="ruleForm" :model="ruleForm" :rules="rules">
				<el-form-item label="账号" prop="account">
					<el-input autocomplete="off" @keyup.enter.native="submitForm('ruleForm')" placeholder="4-15位字母数字下划线" prefix-icon="el-icon-user" v-model="ruleForm.account">
					</el-input>
				</el-form-item>
				<el-form-item label="昵称" prop="nickName">
					<el-input autocomplete="off" placeholder="可选，默认同账号" prefix-icon="el-icon-postcard" v-model="ruleForm.nickName">
					</el-input>
				</el-form-item>
				<el-form-item label="密码" prop="password">
					<el-input autocomplete="off" @keyup.enter.native="submitForm('ruleForm')" placeholder="密码" prefix-icon="el-icon-warning-outline" v-model="ruleForm.password" show-password>
					</el-input>
				</el-form-item>
				<el-form-item label="确认密码" prop="password2">
					<el-input autocomplete="off" @keyup.enter.native="submitForm('ruleForm')" placeholder="再次输入密码" prefix-icon="el-icon-warning-outline" v-model="ruleForm.password2" show-password>
					</el-input>
				</el-form-item>
				<el-form-item>
					<el-button type="primary" @click="submitForm('ruleForm')">注册</el-button>
					<el-button @click="goLogin">返回登录</el-button>
				</el-form-item>
			</el-form>
			<div class="lookout">
				<p>注册成功后将获得学生角色权限，可使用学号相关功能（若与学籍未关联，部分数据可能为空）。</p>
			</div>
		</div>
	</div>
</template>

<script>
import request from '@/utils/request'
import { Notification } from 'element-ui'
export default {
  data () {
    var validateAccount = (rule, value, callback) => {
      var regAccount = /^[a-zA-Z0-9_]{4,15}$/
      if (value === '') {
        callback(new Error('账号不能为空!'))
      } else if (regAccount.test(value) === false) {
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
    var validatePassword2 = (rule, value, callback) => {
      if (value === '') {
        callback(new Error('请再次输入密码!'))
      } else if (value !== this.ruleForm.password) {
        callback(new Error('两次输入密码不一致!'))
      } else {
        callback()
      }
    }
    return {
      registerLoading: false,
      ruleForm: {
        account: '',
        nickName: '',
        password: '',
        password2: ''
      },
      rules: {
        account: [{
          validator: validateAccount,
          trigger: 'blur'
        }],
        password: [{
          validator: validatePassword,
          trigger: 'blur'
        }],
        password2: [{
          validator: validatePassword2,
          trigger: 'blur'
        }]
      }
    }
  },
  methods: {
    goLogin () {
      this.$router.push('/login')
    },
    submitForm (formName) {
      this.$refs[formName].validate((valid) => {
        if (valid) {
          this.registerLoading = true
          const payload = {
            username: this.ruleForm.account,
            password: this.ruleForm.password
          }
          const nick = (this.ruleForm.nickName || '').trim()
          if (nick) {
            payload.nickName = nick
          }
          request.post('/api/auth/register', payload).then(res => {
            this.registerLoading = false
            if (!res || res.data === undefined) return
            if (res.data.status === -1) return
            Notification({
              type: 'success',
              message: res.data.msg
            })
            this.$router.push('/login')
          }).catch(() => {
            this.registerLoading = false
          })
        } else {
          return false
        }
      })
    }
  }
}
</script>

<style lang="less">
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
			min-height: 460px;
			top: 80px;
			right: 12%;
			background: white;
			border-radius: 10px;
			box-shadow: 3px 3px 10px 2px #0b9243;
			padding: 10px;

			.title {
				color: #777;
				font-size: 28px;
				text-align: center;
				margin: 28px 0px;
			}

			.lookout {
				color: #777;
				padding: 12px 20px 20px;
				font-size: 13px;
			}

			.el-input {
				width: 85%;
			}

		}
	}
</style>
