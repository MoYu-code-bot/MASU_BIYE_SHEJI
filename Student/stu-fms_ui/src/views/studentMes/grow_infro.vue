<template>
  <div>
    <div style="padding: 20px;">
      <!--      查询部分-->
      <div id="query" >
        <el-button type="primary" @click="openDetail({})" :loading="insertLoading">
          <i class="el-icon-plus"></i>新增</el-button>
        <span style="width: 10px;height: 2px;display: inline-block"></span>
        <el-popconfirm
          title="确定删除这些数据吗？"
          @onConfirm="deleteBatch">
          <el-button slot="reference" type="danger" @click="" :loading="deleteBatchLoading">
            <i class="el-icon-delete"></i>批量删除</el-button>
        </el-popconfirm>
        <span style="width: 10px;height: 2px;display: inline-block"></span>
        <div style="width:30%;float: right">
          <el-input @keyup.enter.native="search" placeholder="请输入学号" v-model="searchValue" class="input-with-select">
            <el-button @click="search" slot="append" icon="el-icon-search" :loading="searchLoading"></el-button>
          </el-input>
        </div>
      </div>
    <div class="growMes">
      <el-table
        :data="tableData"
        v-loading="tableLoading"
        @selection-change="selectionChange"
        height="370"
        style="width: 100%">
        <el-table-column type="selection" width="45">
        </el-table-column>
        <el-table-column
          prop="stuId"
          label="学号"
          width="180">
          <template slot-scope="scope">
            <el-link type="primary" @click="linkToStuInfo(scope.row)">{{scope.row.stuId}}</el-link>
          </template>
        </el-table-column>
        <el-table-column
          prop="company"
          label="就职单位"
          width="200">
        </el-table-column>
        <el-table-column
          prop="inTime"
          label="入职时间"
          width="160">
        </el-table-column>
        <el-table-column
          prop="salary"
          label="薪资(元)"
          width="120">
        </el-table-column>
        <el-table-column
          prop="auditStatusCode"
          label="审核状态">
          <template slot-scope="scope">
            <span>{{$dict.getValue(scope,'audit_status',scope.row.auditStatusCode)}}</span>
          </template>
        </el-table-column>
        <el-table-column
          fixed="right"
          label="操作"
          width="250">
          <template slot-scope="scope">
            <el-button @click="openDetail(scope.row)" type="primary" size="small">修改</el-button>
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
    </div>
    <!--修改信息对话框-->
    <div class="edit">
      <el-dialog @close="closeDetail" title="就业发展信息" :visible.sync="showDetail">
        <el-form :model="detail">
          <el-form-item label="学号" :labewidth="formLabelWidth">
            <el-input v-model="detail.stuId" autocomplete="off"></el-input>
          </el-form-item>
          <el-form-item label="就职单位" :labewidth="formLabelWidth">
            <el-input v-model="detail.company" autocomplete="off"></el-input>
          </el-form-item>
          <el-form-item label="薪资" :labewidth="formLabelWidth">
            <el-input type="number" v-model="detail.salary" autocomplete="off"></el-input>
          </el-form-item>
          <el-form-item label="入职时间" :labewidth="formLabelWidth">
            <el-date-picker
              v-model="detail.inTime"
              type="date"
              placeholder="选择日期"
              style="width: 207.27px">
            </el-date-picker>
          </el-form-item>
        </el-form>
        <div slot="footer" class="dialog-footer">
          <el-button @click="showDetail = false">取 消</el-button>
          <el-button v-if="detail.id" type="primary" :loading="updateLoading" @click="updateRow">保存</el-button>
          <el-button v-if="!detail.id" type="primary" :loading="insertLoading" @click="insertRow">新增</el-button>
        </div>
      </el-dialog>
    </div>
  </div>
  </div>
</template>

<script>
  import request from '../../utils/request'
  import { Notification } from 'element-ui'

  export default {
    name: 'grow_infro',
    data () {
      return {
        tableData: [],
        tableLoading:false,
        updateLoading:false,
        insertLoading:false,
        deleteBatchLoading:false,
        searchLoading:false,
        searchValue:"",
        selectedRows:[],
        currentPage:1,
        pageSize:10,
        total:0,
        showDetail:false, //是否显示详情弹框
        detail:{}, //显示详情的对象
        showDeleteConfirm:false,
        formLabelWidth: '120px',
        form: {
          depart:'',
          desc:'',
          work_time:''
        },
        form1:{},
        ruleForm:{
          depart:'',
          desc:'',
          work_time:''
        },
        rules:{
          depart:[
            {required:true,message:'请输入就职单位',trigger:'blur'}
          ],
          desc:[
            {required:true,message:'请输入任职情况',trigger:'blur'}
          ],
          work_time:[
            {required:true,message:'请输入已工作时间',trigger:'blur'}
          ],
        }
      }
    },
    mounted () {
      this.search();
    },
    methods: {
      linkToStuInfo(row){
        this.$router.push({
          path:'/basic',
          query:{
            stuId:row.stuId
          }
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
      search () {
        this.tableLoading=true;
        request.get("/api/employ/search",{
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
      //打开详情弹框
      openDetail(row){
        this.showDetail=true;
        this.detail={...row};
      },
      //关闭详情弹框
      closeDetail(row){
        this.showDetail=false;
        this.detail={};
      },
      deleteRow(row){
        this.tableLoading=true;
        request.post("/api/employ/delete",{
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
        request.post("/api/employ/deleteBatch",{ids}).then(res=>{
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
        request.post("/api/employ/insert",{...this.detail}).then(res=>{
          this.insertLoading=false;
          if(res.data.status==0){
            Notification({
              type: 'success',
              message: '新增成功'
            })
            this.closeDetail()
            //刷新表格
            this.search();
          }
        });
      },
      updateRow(){
        this.updateLoading=true;
        request.post("/api/employ/update",{...this.detail}).then(res=>{
          this.updateLoading=false;
          Notification({
            type: 'success',
            message: '保存成功'
          })
          this.closeDetail();
          //刷新表格
          this.search();
        });
      },
      selectionChange(val){
        this.selectedRows=val;
      },
      subStr(str,num){
        if(str&&str.length>num){
          str=str.substring(0,num)+"...";
        }
        return  str;
      }
    }
  }
</script>

<style scoped>
  .header{
    padding-top: 20px;
  }
  .header>.el-button{
    margin-left: 10px;
  }
</style>
