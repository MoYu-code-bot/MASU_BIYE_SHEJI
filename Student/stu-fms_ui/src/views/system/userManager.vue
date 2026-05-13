<template>
	<div id="userManager" >
		<!-- 多条件查询区域 -->
		<div id="query">
      <el-button type="primary" @click="showInsertUser=true" :loading="insertLoading">
        <i class="el-icon-plus"></i>新增</el-button>
      <span style="width: 10px;height: 2px;display: inline-block"></span>
      <el-popconfirm
        title="确定删除这些用户吗？"
        @onConfirm="deleteUserBatch">
        <el-button slot="reference" type="danger" @click="" :loading="deleteBatchLoading">
          <i class="el-icon-delete"></i>批量删除</el-button>
      </el-popconfirm>

      <div style="width:30%;float: right">
        <el-input @keyup.enter.native="search" placeholder="请输入内容" v-model="searchValue" class="input-with-select">
          <el-button @click="search" slot="append" icon="el-icon-search" :loading="searchLoading"></el-button>
        </el-input>
      </div>
		</div>
		<!-- 表格部分 -->
		<div id="table" >
			<el-table
        @selection-change="selectionChange"
        v-loading="tableLoading"
        height="370"
        :data="tableData"
        border
        style="width: 100%">
				<el-table-column type="selection" width="55"></el-table-column>
				<el-table-column  prop="username" label="用户名"></el-table-column>
				<el-table-column  prop="password" label="密码">
          <template slot-scope="psw">
            <span>******</span>
          </template>
        </el-table-column>
        <el-table-column  prop="roleName" label="角色"></el-table-column>
        <el-table-column prop="avatar"  label="头像" >
          <template slot-scope="scope">
            <el-avatar shape="square" :size="30" :src="scope.row.avatar"></el-avatar>
          </template>
        </el-table-column>
        <el-table-column  prop="realName" label="真实姓名" ></el-table-column>
				<el-table-column  prop="nickName" label="昵称" ></el-table-column>
				<el-table-column  prop="mobile" label="电话"></el-table-column>
        <el-table-column  prop="gender" label="性别"></el-table-column>
				<el-table-column fixed="right" label="操作" width="210">
					<template slot-scope="scope">
						<el-row>
							<el-col :span="8">
								<el-button @click="detail(scope.row)" type="primary" size="mini">
									详情</el-button>
							</el-col>
							<el-col :span="8">
								<el-button @click="changePassword(scope.row)" type="primary" size="mini">
									修改</el-button>
							</el-col>
							<el-col :span="8">
								<el-popconfirm
								  title="确定删除这个用户吗？"
								 v-model="showDeleteConfirm"
								 @onConfirm="deleteUser(scope.row)">
								  <el-button slot="reference" type="danger" size="mini">删除</el-button>
								</el-popconfirm>
							</el-col>
						</el-row>
					</template>
				</el-table-column>
			</el-table>
<!--      分页-->
      <el-pagination
        @size-change="handleSizeChange"
        @current-change="handleCurrentChange"
        :current-page="currentPage"
        :page-sizes="[10,20,50]"
        :page-size="pageSize"
        layout="total, sizes, prev, pager, next, jumper"
        :total="total">
      </el-pagination>
		</div>
		<el-dialog title="修改" :visible.sync="showDialog" width="40%" :before-close="closeDialog">
			<el-form @submit.native.prevent label-width="80px" v-model="myDetail" >
				<el-form-item label="密码："  required>
					<el-input @keyup.enter.native="comfirmDialog" placeholder="新密码" prefix-icon="el-icon-warning-outline"
					 v-model="myDetail.password" show-password>
					</el-input>
				</el-form-item>
        <el-form-item label="角色：" required>
          <el-select v-model="myDetail.roleId" placeholder="请选择">
            <el-option
              v-for="item in roles"
              :key="item.id"
              :label="item.roleName"
              :value="item.id">
            </el-option>
          </el-select>
        </el-form-item>
			</el-form>
			<span slot="footer" class="dialog-footer">
				<el-button @click="closeDialog">取 消</el-button>
				<el-button type="primary" :loading="updateLoading" @click="comfirmDialog">确认修改</el-button>
			</span>
		</el-dialog>
<!--    详情-->
    <el-dialog width="65%" title="详情" :visible.sync="showUserDetail">
      <el-form :inline="true" :model="detailForm">
        <el-form-item label="用户名:">
          <el-input v-model="detailForm.username"  readonly></el-input>
        </el-form-item>
        <el-form-item label="密码:">
          <el-input v-model="detailForm.password" readonly></el-input>
        </el-form-item>
        <el-form-item label="昵称:">
          <el-input v-model="detailForm.nickName" readonly></el-input>
        </el-form-item>
        <el-form-item label="真实姓名：">
          <el-input v-model="detailForm.password" readonly></el-input>
        </el-form-item>
        <el-form-item label="电话：">
          <el-input v-model="detailForm.mobile" readonly></el-input>
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button type="primary" @click="showUserDetail = false">知道了</el-button>
      </div>
    </el-dialog>
    <!--    新增-->
    <el-dialog width="35%" title="新增" :visible.sync="showInsertUser">
      <el-form  :model="insertForm">
        <el-collapse v-model="activeName" >
          <el-collapse-item title="必填项" name="必填">
            <el-form-item label="用户名:" required>
              <el-input placeholder="请输入用户名" v-model="insertForm.username"  ></el-input>
            </el-form-item>
            <el-form-item label="密码:" required >
              <el-input placeholder="请输入密码" type="password" v-model="insertForm.password" ></el-input>
            </el-form-item>
            <el-form-item label="真名：" required>
              <el-input placeholder="请输入真实姓名" v-model="insertForm.realName" ></el-input>
            </el-form-item>
            <el-form-item label="电话：" required>
              <el-input v-model="insertForm.mobile" ></el-input>
            </el-form-item>
          </el-collapse-item>
          <el-collapse-item title="选填项" name="选填">
            <el-form-item label="头像:">
              <el-upload
                class="upload-demo"
                action="/api/upload/img"
                :file-list="fileList"
                :multiple="false"
                @on-success="uploadSuccess"
                @on-error="uploadError"
                ref="upload"
                name="file"
                :limit="1"
                :auto-upload="false"
                list-type="picture">
                <el-button size="small" type="primary">点击上传</el-button>
                <div slot="tip" class="el-upload__tip">只能上传jpg/png文件，且不超过500kb</div>
              </el-upload>
            </el-form-item>
            <el-form-item label="昵称:">
              <el-input v-model="insertForm.nickName" readonly></el-input>
            </el-form-item>
            <el-form-item label="邮箱:">
              <el-input v-model="insertForm.email" readonly></el-input>
            </el-form-item>
            <el-form-item label="性别:">
              <el-radio-group v-model="insertForm.gender">
                <el-radio label="0">女</el-radio>
                <el-radio label="1">男</el-radio>
              </el-radio-group>
            </el-form-item>
            <el-form-item label="微信:">
              <el-input v-model="insertForm.wechat" readonly></el-input>
            </el-form-item>
            <el-form-item label="生日:">
              <el-input v-model="insertForm.birthday" readonly></el-input>
            </el-form-item>
          </el-collapse-item>
        </el-collapse>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button @click="showInsertUser = false">取 消</el-button>
        <el-button type="primary" @click="insertUser" :loading="insertLoading">提 交</el-button>
      </div>
    </el-dialog>
	</div>
</template>

<script>
import request from '@/utils/request'
import { Notification } from 'element-ui'
export default {
  name: 'userManager',
  data () {
    var validatePassword = (rule, value, callback) => {
      if (value === '') {
        callback(new Error('密码不能为空!'))
      } else {
        callback()
      }
    }
    return {
      squareUrl: "https://cube.elemecdn.com/9/c2/f0ee8a3c7c9638a54940382568c9dpng.png",
      activeName:'必填',
      showInsertUser:false,
      insertForm:{},
      newPassword:"",
      detailForm:{},
      showUserDetail:false,
      total:0,
      pageSize:10,
      currentPage:1,
      fileList: [],
      queryModel: {
        account: null
      },
      tableData: [],
      editInfo: {},
      showDialog: false,
      selectedRows:[],
      showDeleteConfirm: false,
      newPassword: '',
      rules: {
        password: [{ validator: validatePassword, trigger: 'blur' }]
      },
      updateLoading:false,
      tableLoading:false,
      insertLoading:false,
      deleteBatchLoading:false,
      searchLoading:false,
      searchValue:"",
      roles:[],
      myDetail:{}
    }
  },
  mounted () {
    this.search();
    this.requestAllRoles();
  },
  methods: {
    requestAllRoles(){
      request.get("/api/role/all").then(res=>{
        this.roles=res.data.data;
      });
    },
    uploadError(err, file, fileList){
      console.log(err, file, fileList)
    },
    uploadSuccess(response, file, fileList){
      console.log(response, file, fileList)
    },
    //搜索
    search(){
      this.tableLoading=true;
      request.get("/api/user/vo/search",{
        params:{
          searchValue:this.searchValue,
          page:this.currentPage,
          size:this.pageSize
        }
      }).then(res=>{
        this.tableLoading=false;
        this.total=res.data.data.total;
        this.tableData=res.data.data.records;
      });
    },
    handleCurrentChange(curPage){
      this.currentPage=curPage;
      this.search();
    },
    handleSizeChange(size){
      this.pageSize=size;
      this.search();
    },
    query () {
      console.log(this.queryModel)
    },
    //确认修改
    comfirmDialog () {
      this.updateLoading=true;
      request.post("/api/user/vo/update",this.myDetail).then(res=>{
        this.updateLoading=false;
        this.showDialog = false
        if(res.data.status==0){
          Notification({
            type: 'success',
            message: '修改成功'
          });
          //刷新数据
          this.search();
        }
      });
    },
    closeDialog () {
      this.showDialog = false
    },
    // 删除用户事件
    deleteUser (row) {
      request.post("/api/user/delete",{
        id:row.id
      }).then(res=>{
        if(res.data.status==0){
          Notification({
            type: 'success',
            message: '删除成功'
          })
          //刷新数据
          this.search();
        }
      });
    },
    //批量删除
    deleteUserBatch(){
      var ids=[];
      for(var i=0;i<this.selectedRows.length;i++){
        ids.push(this.selectedRows[i].id);
      }
      if(ids.length==0){
        return Notification({
          type: 'info',
          message: '请选择需要删除用户'
        });
      }
      this.deleteBatchLoading=true
      request.post("/api/user/deleteBatch",{ids}).then(res=>{
        this.deleteBatchLoading=false;
        if(res.data.status==0){
          Notification({
            type: 'success',
            message: '批量删除成功'
          })
          //刷新数据
          this.search();
        }
      });
    },
    //新增用户
    insertUser(){
      this.insertLoading=true;
      //上传图片
      this.$refs.upload.submit();
      request.post("/api/user/insert",{...this.insertForm}).then(res=>{
        this.insertLoading=false;
        this.showInsertUser=false;
        if(res.data.status==0){
          Notification({
            type: 'success',
            message: '新增成功'
          })
          //刷新数据
          this.search();
        }
      });
    },
    // 查看详情事件
    detail (row) {
      this.showUserDetail=true;
      this.detailForm=row;
    },
    // 改变密码事件
    changePassword (row) {
      // 显示dialog
      this.showDialog = true
      this.myDetail={...row};
    },
    selectionChange(val){
      this.selectedRows=val;
    },
    handleRemove(file, fileList) {
      console.log(file, fileList);
    },
    handlePreview(file) {
      console.log(file);
    }
  }
}
</script>

<style scoped lang="less">
	#userManager {
		padding: 20px;

		#query {
			padding-bottom: 5px;
		}

		.name {
			color: red;
			font-weight: 700;
		}
	}
</style>
