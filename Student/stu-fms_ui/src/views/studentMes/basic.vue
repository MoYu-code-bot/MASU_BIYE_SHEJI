<template>
  <div>
    <!--tab选项卡-->
    <el-tabs type="border-card" v-model="activeName">
      <!--学生总的信息-->
      <el-tab-pane v-if="$shiro.hasPermission(this,'stuInfoManager')" name="all">
        <span slot="label"><i class="el-icon-info"></i>学生信息</span>
<!--        <basic-all @all="onFn"></basic-all>-->
        <el-button v-loading="insertLoading" type="primary" size="medium" icon="el-icon-plus" @click="add">新增</el-button>
        <span style="width: 10px;height: 2px;display: inline-block"></span>
        <el-popconfirm
          title="确定删除这些数据吗？"
          @onConfirm="deleteBatch">
          <el-button slot="reference" type="danger" @click="" :loading="deleteBatchLoading">
            <i class="el-icon-delete"></i>批量删除</el-button>
        </el-popconfirm>
        <div style="width:20%;float: right">
          <el-input @keyup.enter.native="search" placeholder="请输入学号..." v-model="searchValue" class="input-with-select">
            <el-button @click="search" slot="append" icon="el-icon-search" :loading="searchLoading"></el-button>
          </el-input>
        </div>
        <!--表格-->
        <el-table
          :data="tableData"
          v-loading="tableLoading"
          height="370"
          @selection-change="selectionChange">
          <el-table-column type="selection" width="45">
          </el-table-column>
          <el-table-column prop="stuId" label="学号" >
          </el-table-column>
          <el-table-column prop="address" label="家庭住址" >
          </el-table-column>
          <el-table-column prop="political" label="政治面貌">
          </el-table-column>
          <el-table-column prop="college" label="学院名称">
          </el-table-column>
          <el-table-column prop="prof" label="专业名称">
          </el-table-column>
          <el-table-column prop="stuClass" label="行政班级" >
          </el-table-column>
          <el-table-column prop="inTime" label="入学年月" >
            <template slot-scope="scope">
              <i class="el-icon-time"></i>
              <span style="margin-left: 10px">{{ scope.row.inTime }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="outTime" label="毕业年月">
            <template slot-scope="scope">
              <i class="el-icon-time"></i>
              <span style="margin-left: 10px">{{ scope.row.outTime }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="academic" label="毕业学历">
          </el-table-column>
          <el-table-column prop="degree" label="毕业学位" >
          </el-table-column>
          <el-table-column
            fixed="right"
            label="操作"
            width="250">
            <template slot-scope="scope">
              <el-button size="mini" type="infor" icon="el-icon-edit" @click="edit(scope.row)">查看</el-button>
              <span style="width: 10px;height: 2px;display: inline-block"></span>
              <el-popconfirm
                title="确定删除这个用户吗？"
                v-model="showDeleteConfirm"
                @onConfirm="deleteRow(scope.row)">
                <el-button slot="reference" type="danger" size="small">删除</el-button>
              </el-popconfirm>
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
      </el-tab-pane>
      <!--学生新增-->
      <el-tab-pane name="add">
        <span slot="label"><i class="el-icon-edit-outline"></i>详情</span>
        <div class="container" style="margin-top: 20px">
          <div style="padding-top: 20px">
            <el-form ref="form" :model="form" label-width="100px" class="demo-form-inline" :inline="true">
              <el-form-item label="学号">
                <el-input v-model="form.stuId"></el-input>
              </el-form-item>
              <el-form-item label="家庭住址">
                <el-input v-model="form.address"></el-input>
              </el-form-item>
              <el-form-item label="政治面貌">
                <el-input v-model="form.political"></el-input>
              </el-form-item>
              <el-form-item label="入学年月">
                <el-date-picker
                  v-model="form.inTime"
                  type="date"
                  placeholder="选择日期"
                  style="width: 207.27px">
                </el-date-picker>
              </el-form-item>
              <el-form-item label="毕业年月">
                <el-date-picker
                  v-model="form.outTime"
                  type="date"
                  placeholder="选择日期"
                  style="width: 207.27px">
                </el-date-picker>
              </el-form-item>
              <el-form-item label="学院名称">
                <el-input v-model="form.college"></el-input>
              </el-form-item>
              <el-form-item label="专业名称">
                <el-input v-model="form.prof"></el-input>
              </el-form-item>
              <el-form-item label="行政班级">
                <el-input v-model="form.stuClass"></el-input>
              </el-form-item>
              <el-form-item label="毕业学历">
                <el-select v-model="form.academic" clearable placeholder="请选择">
                  <el-option
                    v-for="item in options"
                    :key="item.value"
                    :label="item.label"
                    :value="item.value">
                  </el-option>
                </el-select>
              </el-form-item>
              <el-form-item label="毕业学位">
                <el-select v-model="form.degree" clearable placeholder="请选择">
                  <el-option
                    v-for="item in options1"
                    :key="item.value"
                    :label="item.label"
                    :value="item.value">
                  </el-option>
                </el-select>
              </el-form-item>
              <el-form-item style="margin-left: 900px">
                <el-button v-if="form.id" type="primary" @click="updateRow" style="margin-right: 30px">保存</el-button>
                <el-button v-if="!form.id" type="primary" @click="insertRow" style="margin-right: 30px">新增</el-button>
                <el-button style="margin-right: 30px" @click="onreset">重置</el-button>
              </el-form-item>
            </el-form>
          </div>
        </div>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script>
import request from '../../utils/request'
import { Notification } from 'element-ui'
export default {
  name: 'basic',
  data () {
    return {
      showDeleteConfirm:false,
      activeName: 'all',
      selectIds: [], // 被选的的ids
      tableData: [],
      tableLoading:false,
      updateLoading:false,
      insertLoading:false,
      deleteBatchLoading:false,
      searchLoading:false,
      searchValue:"",
      selectedRows:[],
      form: {},
      value: '',
      value1: '',
      options: [{
        value: '专科',
        label: '专科'
      }, {
        value: '本科',
        label: '本科'
      }, {
        value: '硕士研究生',
        label: '硕士研究生'
      }, {
        value: '博士研究生',
        label: '博士研究生'
      }],
      options1: [{
        value: '学士',
        label: '学士'
      }, {
        value: '硕士',
        label: '硕士'
      }, {
        value: '博士',
        label: '博士'
      }],
      currentPage:1,
      pageSize:10,
      total:0,
    }
  },
  mounted () {
    console.log("坚定",this.$shiro.hasPermission(this,"menu:archives-manage"));
    //取值
    var stuId=this.$route.query.stuId;
    if(stuId){
      this.loadStuInfo(stuId);
    }
    //加载自己的信息
    this.loadMySelfInfo();
    this.search()
  },
  methods: {
    loadMySelfInfo(){
      if(!this.$shiro.hasPermission(this,'stuInfoManager')){
        var uid=this.$store.state.user.id;
        if(uid){
          //只加载自己的信息
          request.get("/api/stu_info/getByUid",{
            params:{uid:uid}
          }).then(res=>{
            if(res.data.status==0&&res.data.data){
              this.edit(res.data.data);
            }
          });
        }else{
          console.log("不加载任何数据");
        }
      }
    },
    loadStuInfo(stuId){
      if(this.$shiro.hasPermission(this,'stuInfoManager')){
        //加载其他人的信息
        request.get("/api/stu_info/getByStuId",{
          params:{stuId:stuId}
        }).then(res=>{
          if(res.data.status==0){
            this.edit(res.data.data);
          }
        });
      }
    },
    //新增
    add(){
      this.activeName = 'add';
      this.form = {};
      this.value ='';
      this.value1 = '';
    },
    //查看详情
    edit(mes){
      this.activeName = 'add';
      this.form = mes;
      this.value = mes.academic;
      this.value1 = mes.degree;
    },
    handleCurrentChange(curPage){
      this.currentPage=curPage;
      this.search();
    },
    handleSizeChange(size){
      this.pageSize=size;
      this.search();
    },
    //搜索
    search () {
      this.tableLoading=true;
      request.get("/api/stu_info/search",{
        params:{
          searchValue:this.searchValue,
          page:this.currentPage,
          size:this.pageSize
        }
      }).then(res=>{
        this.total=res.data.data.total;
        this.tableData=res.data.data.records;
        this.tableLoading=false;
      });
    },
    deleteRow(row){
      this.tableLoading=true;
      request.post("/api/stu_info/delete",{
        id:row.id
      }).then(res=>{
        if(res.data.status==0){
          this.tableLoading=false;
          Notification({
            type: 'success',
            message: '删除成功'
          })
          //刷新表格
          this.search();
        }
      });
    },
    deleteBatch(){
      var ids=[];
      for(var i=0;i<this.selectedRows.length;i++){
        ids.push(this.selectedRows[i].id);
      }
      if(ids.length==0){
        return Notification({
          type: 'info',
          message: '请选择需要删除的数据'
        });
      }
      this.deleteBatchLoading=true
      request.post("/api/stu_info/deleteBatch",{ids}).then(res=>{
        this.deleteBatchLoading=false;
        if(res.data.status==0){
          Notification({
            type: 'success',
            message: '批量删除成功'
          })
          //刷新表格
          this.search();
        }
      });
    },
    insertRow(){
      this.insertLoading=true;
      request.post("/api/stu_info/insert",{...this.form}).then(res=>{
        this.insertLoading=false;
        if(res.data.status==0){
          Notification({
            type: 'success',
            message: '新增成功'
          })
          //刷新表格
          this.search();
        }
      });
    },
    updateRow(){
      this.updateLoading=true;
      request.post("/api/stu_info/update",{...this.form}).then(res=>{
        this.updateLoading=false;
        Notification({
          type: 'success',
          message: '保存成功'
        })
        //刷新表格
        this.search();
      });
    },
    selectionChange(val){
      this.selectedRows=val;
    },
    onreset() { // 重置信息，清空已填的所有内容
      this.form = {}
    },
    }
}
</script>

<style scoped>

</style>
