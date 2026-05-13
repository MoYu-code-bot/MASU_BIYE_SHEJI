<template>
    <div style="padding: 20px">
      <!--      查询部分-->
      <div id="query" >
        <el-button type="primary" @click="openDetail({})" :loading="insertLoading">
          <i class="el-icon-plus"></i>新增</el-button>
        <span style="width: 10px;height: 2px;display: inline-block"></span>
        <el-popconfirm
          title="确定删除这些用户吗？"
          @onConfirm="deleteBatch">
          <el-button slot="reference" type="danger" @click="" :loading="deleteBatchLoading">
            <i class="el-icon-delete"></i>批量删除</el-button>
        </el-popconfirm>
        <span style="width: 10px;height: 2px;display: inline-block"></span>
        <el-popover
          placement="bottom"
          width="600"
          offset="10"
          trigger="click">
          <el-form :inline="true" :model="searchForm">
            <el-form-item label="项目名称:" >
              <el-input v-model="searchForm.proName" placeholder="请填写项目名称"></el-input>
            </el-form-item>
            <el-form-item label="项目级别:" >
              <el-select v-model="searchForm.proLevel" placeholder="请选择">
                <el-option label="国际级" value="国际级"></el-option>
                <el-option label="国家级" value="国家级"></el-option>
                <el-option label="省部级" value="省部级"></el-option>
                <el-option label="市、州级" value="市、州级"></el-option>
                <el-option label="校级" value="校级"></el-option>
                <el-option label="院级" value="院级"></el-option>
              </el-select>
            </el-form-item>
            <el-form-item label="项目经费:" >
              <el-input v-model="searchForm.funds" placeholder="请填写项目经费"></el-input>
            </el-form-item>
            <el-form-item label="项目状态:" >
              <el-select v-model="searchForm.status" placeholder="请选择">
                <el-option label="已立项" value="已立项"></el-option>
                <el-option label="完成中" value="完成中"></el-option>
                <el-option label="已完成" value="已完成"></el-option>
              </el-select>
            </el-form-item>
            <el-button type="primary" @click="search"><i class="el-icon-search"></i>查询</el-button>
          </el-form>
          <el-button slot="reference">筛选</el-button>
        </el-popover>
        <div style="width:30%;float: right">
          <el-input @keyup.enter.native="search" placeholder="请输入内容" v-model="searchValue" class="input-with-select">
            <el-button @click="search" slot="append" icon="el-icon-search" :loading="searchLoading"></el-button>
          </el-input>
        </div>
      </div>
      <div class="science">
        <el-table
          :data="tableData"
          v-loading="tableLoading"
          @selection-change="selectionChange"
          height="370"
          style="width: 100%">
          <el-table-column type="selection" width="55"></el-table-column>
          <el-table-column
            prop="stuId"
            label="学号"
            width="100">
            <template slot-scope="scope">
              <el-link type="primary" @click="linkToStuInfo(scope.row)">{{scope.row.stuId}}</el-link>
            </template>
          </el-table-column>
          <el-table-column
            prop="proName"
            label="项目名称"
            width="140">
          </el-table-column>
          <el-table-column
            prop="proLevel"
            label="项目级别"
            width="140">
          </el-table-column>
          <el-table-column
            prop="funds"
            label="项目经费"
            width="140">
          </el-table-column>
          <el-table-column
            prop="status"
            label="项目状态"
            width="140">
          </el-table-column>
          <el-table-column
            prop="auditStatusCode"
            label="审核状态"
            width="140">
            <template slot-scope="scope">
              <span>{{$dict.getValue(scope,'audit_status',scope.row.auditStatusCode)}}</span>
            </template>
          </el-table-column>
          <el-table-column
            prop="descr"
            label="项目简介">
            <template slot-scope="scope">
              <el-popover
                placement="top-start"
                title="项目简介"
                width="400"
                trigger="hover"
                :content="scope.row.descr">
                <span slot="reference">{{subDesc(scope.row.descr)}}</span>
              </el-popover>
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
      <el-dialog title="科研信息" :visible.sync="showDetail">
        <el-form :model="detail">
          <el-form-item label="项目名称" required>
            <el-input v-model="detail.proName" autocomplete="off"></el-input>
          </el-form-item>
          <el-form-item label="项目级别" required>
            <el-select v-model="detail.proLevel" placeholder="请选择">
              <el-option label="国际级" value="国际级"></el-option>
              <el-option label="国家级" value="国家级"></el-option>
              <el-option label="省部级" value="省部级"></el-option>
              <el-option label="市、州级" value="市、州级"></el-option>
              <el-option label="校级" value="校级"></el-option>
              <el-option label="院级" value="院级"></el-option>
            </el-select>
          </el-form-item>
          <el-form-item label="项目经费" required>
            <el-input v-model="detail.funds" autocomplete="off"></el-input>
          </el-form-item>
          <el-form-item label="项目状态" required>
            <el-select v-model="detail.status" placeholder="请选择">
              <el-option label="已立项" value="已立项"></el-option>
              <el-option label="完成中" value="完成中"></el-option>
              <el-option label="已完成" value="已完成"></el-option>
            </el-select>
          </el-form-item>
          <el-form-item label="项目简介" required>
            <el-input   type="textarea" :rows="4" v-model="detail.descr" autocomplete="off"></el-input>
          </el-form-item>
        </el-form>
        <span slot="footer" class="dialog-footer">
            <el-button @click="closeDetail">取 消</el-button>
            <el-button v-if="detail.id" type="primary" :loading="updateLoading" @click="updateRow">保存</el-button>
            <el-button v-if="!detail.id" type="primary" :loading="insertLoading" @click="insertRow">新增</el-button>
          </span>
      </el-dialog>
    </div>
</template>

<script>
import request from '../../utils/request'
import { Notification } from 'element-ui'
export default {
  name: 'scienceMes',
  data () {
    return {
      input: '',
      dialogFormVisible: false,
      formLabelWidth: '120px',
      form: {},
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
      searchForm:{}
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
      request.get("/api/project/search",{
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
      request.post("/api/project/delete",{
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
      request.post("/api/project/deleteBatch",{ids}).then(res=>{
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
      request.post("/api/project/insert",{...this.detail}).then(res=>{
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
    updateRow(row){
      this.updateLoading=true;
      request.post("/api/project/update",{...this.detail}).then(res=>{
        this.updateLoading=false;
        this.closeDetail();
        //刷新表格
        this.search();
      });
    },
    selectionChange(val){
      this.selectedRows=val;
    },
    //提取部分简介
    subDesc(desc){
      return desc.substring(1,15)+"......";
    }
  }
}
</script>

<style scoped>
</style>
