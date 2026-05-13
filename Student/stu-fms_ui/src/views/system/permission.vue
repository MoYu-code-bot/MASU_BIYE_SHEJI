<template>
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
        <el-input @keyup.enter.native="search" placeholder="请输入权限名" v-model="searchValue" class="input-with-select">
          <el-button @click="search" slot="append" icon="el-icon-search" :loading="searchLoading"></el-button>
        </el-input>
      </div>
    </div>
    <div class="weiji">
      <el-table
        :data="tableData"
        v-loading="tableLoading"
        @selection-change="selectionChange"
        height="370"
        style="width: 100%" >
        <el-table-column type="selection" width="55"></el-table-column>
        <el-table-column
          prop="title"
          label="权限简称"
          width="160">
        </el-table-column>
        <el-table-column
          prop="perms"
          label="标识"
          width="300">
        </el-table-column>
        <el-table-column
          prop="type"
          label="权限类型"
          width="160">
        </el-table-column>
        <el-table-column
          prop="descr"
          label="权限描述"
          width="160">
        </el-table-column>
        <el-table-column
          fixed="right"
          label="操作"
          width="250">
          <template slot-scope="scope">
            <el-button @click="openDetail(scope.row)" type="primary" size="small">修改</el-button>
            <span style="width: 10px;height: 2px;display: inline-block"></span>
            <el-popconfirm
              title="确定删除这个数据吗？"
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
      <!--新增对话框-->
      <el-dialog title="权限信息" :visible.sync="showDetail">
        <el-form :model="detail">
          <el-form-item label="权限简称" required :label-width="formLabelWidth">
            <el-input v-model="detail.title" autocomplete="off"></el-input>
          </el-form-item>
          <el-form-item label="标识" required :label-width="formLabelWidth">
            <el-input v-model="detail.perms" autocomplete="off"></el-input>
          </el-form-item>
          <el-form-item label="权限类型" required :label-width="formLabelWidth">
            <el-select v-model="detail.type" placeholder="请选择">
              <el-option label="菜单" value="菜单"></el-option>
              <el-option label="按钮" value="按钮"></el-option>
              <el-option label="url" value="url"></el-option>
              <el-option label="图片" value="图片"></el-option>
              <el-option label="资源" value="资源"></el-option>
            </el-select>
          </el-form-item>
          <el-form-item label="描述" required :label-width="formLabelWidth" >
            <el-input type="textarea" :row="4" v-model="detail.descr" autocomplete="off"></el-input>
          </el-form-item>
        </el-form>
        <div slot="footer" class="dialog-footer">
          <el-button @click="showDetail=false">取 消</el-button>
          <el-button v-if="detail.id" type="primary" :loading="updateLoading" @click="updateRow">保存</el-button>
          <el-button v-if="!detail.id" type="primary" :loading="insertLoading" @click="insertRow">新增</el-button>
        </div>
      </el-dialog>

    </div>
  </div>
</template>

<script>
  import request from '../../utils/request'
  import { Notification } from 'element-ui'

  export default {
    name: 'weijiMes',
    data () {
      return {
        currentPage3: 1,
        selectIds:[],//被选的ids
        searchData:'',//搜索数据
        dialogFormVisible:false,
        formLabelWidth: '120px',
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
        form:{},
        ruleForm:{
          name:'',
          time:'',
          desc:''
        },
        rules:{
          name:[
            {required:true,message:'请输入违纪名称',trigger:'blur'}
          ],
          time:[
            {required:true,message:'请输入违纪时间',trigger:'blur'}
          ],
          desc:[
            {required:true,message:'请输入违纪描述',trigger:'blur'}
          ],
        }
      }
    },
    mounted () {
      this.search();
    },
    methods: {
      //提交新增信息
      submit(formName){
        this.dialogFormVisible = false;
        this.$refs[formName].validate((valid)=>{
          if (valid){
            alert('提交成功！');
            this.tableData.push(this.form); //添加数据
          }else{
            this.dialogFormVisible = true;
            console.log('error submit!');
            return false;
          }
        });
        //提交成功后，将form的数据置为空
        //this.form.name = ''
        //this.form.time = ''
      },
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
        request.get("/api/permission/search",{
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
        request.post("/api/permission/delete",{
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
        request.post("/api/permission/deleteBatch",{ids}).then(res=>{
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
        request.post("/api/permission/insert",{...this.detail}).then(res=>{
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
        request.post("/api/permission/update",{...this.detail}).then(res=>{
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
