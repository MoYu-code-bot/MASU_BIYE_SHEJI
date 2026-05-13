<template>
  <div style="padding: 20px;">
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
      <div style="width:30%;float: right">
        <el-input @keyup.enter.native="search" placeholder="请输入内容" v-model="searchValue" class="input-with-select">
          <el-button @click="search" slot="append" icon="el-icon-search" :loading="searchLoading"></el-button>
        </el-input>
      </div>
    </div>
    <!--显示表格-->
    <div class="activity">
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
          width="200">
          <template slot-scope="scope">
            <el-link type="primary" @click="linkToStuInfo(scope.row)">{{scope.row.stuId}}</el-link>
          </template>
        </el-table-column>
        <el-table-column
          prop="activityName"
          label="活动/竞赛名称"
          width="200">
        </el-table-column>
        <el-table-column
          prop="activityLevel"
          label="活动/竞赛等级"
          width="180">
        </el-table-column>
        <el-table-column
          prop="activityType"
          label="活动/竞赛类型"
          width="180">
        </el-table-column>
        <el-table-column
          prop="activityTime"
          label="参与/获得时间"
          width="160">
        </el-table-column>
        <el-table-column
          prop="img"
          label="参与/获得证明"
          width="300"
          height="300">
          <template slot-scope="scope">
            <el-link type="primary" @click="openMutipleImg(scope.row.img)">查看证明</el-link>
          </template>
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
    <!--新增信息弹出框-->

    <el-dialog @close="closeDetail" title="活动/竞赛信息" :visible.sync="showDetail">
      <el-form :model="detail">
        <el-form-item label="学号:" required :label-width="formLabelWidth" >
          <el-input v-model="detail.stuId" autocomplete="off"></el-input>
        </el-form-item>
        <el-form-item label="活动/竞赛名称" required :label-width="formLabelWidth">
          <el-input v-model="detail.activityName" autocomplete="off"></el-input>
        </el-form-item>
        <el-form-item label="活动/竞赛类型" required :label-width="formLabelWidth">
          <el-select v-model="detail.activityType" clearable placeholder="请选择">
            <el-option
              v-for="item in options"
              :key="item.value"
              :label="item.label"
              :value="item.value">
            </el-option>
          </el-select>
        </el-form-item>
        <el-form-item label="活动/竞赛等级" required :label-width="formLabelWidth" >
          <el-select v-model="detail.activityLevel" clearable placeholder="请选择">
            <el-option label="国际级" value="国际级"></el-option>
            <el-option label="国家级" value="国家级"></el-option>
            <el-option label="省部级" value="省部级"></el-option>
            <el-option label="市、州级" value="市、州级"></el-option>
            <el-option label="校级" value="校级"></el-option>
            <el-option label="院级" value="院级"></el-option>
          </el-select>
        </el-form-item>
        <el-form-item label="参与/获得时间" required :label-width="formLabelWidth">
          <el-date-picker
            v-model="detail.activityTime"
            type="date"
            placeholder="选择日期"
            style="width: 207.27px">
          </el-date-picker>
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button @click="dialogFormVisible = false">取 消</el-button>
        <el-button v-if="detail.id" type="primary" :loading="updateLoading" @click="updateRow">保存</el-button>
        <el-button v-if="!detail.id" type="primary" :loading="insertLoading" @click="insertRow">新增</el-button>
        <!--        <el-button type="primary" @click="submit('ruleForm')">提交</el-button>-->
      </div>
    </el-dialog>

    <el-dialog :visible.sync="dialogVisible">
      <img width="100%" :src="dialogImageUrl" alt="">
    </el-dialog>
    <el-dialog title="证明材料"
     :visible.sync="showMutipleImg"
    @close="closeMutipleImg">
      <span v-if="imgUrls.length==0">未上传证明材料</span>
      <template v-if="imgUrls.length>0">
        <el-image v-for="item in imgUrls"
          style="width: 100px; height: 100px"
          :src="item"
          :lazy="true"
          :preview-src-list="[item]">
        </el-image>
      </template>
    </el-dialog>
  </div>
</template>

<script>
  import request from '../../utils/request'
  import { Notification } from 'element-ui'

  export default {
    name: 'activity',
    data () {
      return {
        showMutipleImg:false,
        imgUrls:[], //多图显示的src
        imgsList:[],
        //验证相关
        ruleForm: {
          name: '',
          time: '',
          type: [],
          level: '',
          img: ''
        },
        //验证信息
        rules:{
          stuId:[
            { required: true,message:'请输入学号',trigger:'blur' }
          ]
        },
        url: '',
        dialogImageUrl: '',
        dialogVisible: false,
        disabled: false,
        input: '',
        options: [{
          value: '学术竞赛',
          label: '学术竞赛'
        }, {
          value: '体育活动',
          label: '体育活动'
        }, {
          value: '校园活动',
          label: '校园活动'
        }, {
          value: '道德奖励',
          label: '道德奖励'
        }],
        types: '',
        dialogFormVisible: false,
        formLabelWidth: '180px',
        form: {
          verify_id: '×'
        },
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
      //打开多图显示
      //urls:以分号分割的图片链接
      openMutipleImg(urls){
        this.showMutipleImg=true;
        if(urls){
          var array=urls.split(';');
          for(var i=0;i<array.length;i++){
            array[i]=array[i].replace(/^\s+|\s+$/g,"");
            if(array[i].length>0){
              this.imgUrls.push(array[i]);
              this.detail.img+=";"+array[i];
            }
          }
        }
      },
      closeMutipleImg(){
        this.showMutipleImg=false;
        this.imgUrls=[];
      },
      uploadSuccess(res){
        if(!this.detail.img){
          this.detail.img="";
        }
        this.detail.img+=";"+res.data;
      },
      handleRemove (url) {
        //删除预览的图片
        var newList=[];
        //重新生成数据里面的
        this.detail.img="";
        for(var i=0;i<this.imgsList.length;i++){
          if(url!=this.imgsList[i].url){
            newList.push(this.imgsList[i]);
            this.detail.img+=";"+this.imgsList[i].url;
          }
        }
        this.imgsList=newList;
      },
      handlePictureCardPreview (file) {
        this.dialogImageUrl = file.url
        this.dialogVisible = true
        this.url = file.url  //将图片路径存入变量
      },
      // submit (formName) {
      //   this.form.type = this.types
      //   this.form.img = this.url  //将图片路径赋值给form.img
      //   this.$refs[formName].validate((valid)=>{
      //     if (valid){
      //       console.log(this.form);
      //     }else{
      //       console.log('error submit!')
      //       return false;
      //     }
      //   });
      // },
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
        request.get("/api/activity/search",{
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
        if(this.detail.id){
          if(this.detail.img && this.detail.img.length>0){
            var array=this.detail.img.split(";");
            for(var i=0;i<array.length;i++){
              if(array[i].length>0){
                this.imgsList.push({url:array[i]});
              }
            }
          }
        }
      },
      //关闭详情弹框
      closeDetail(row){
        this.showDetail=false;
        this.detail={};
        this.imgsList=[];
      },
      deleteRow(row){
        this.tableLoading=true;
        request.post("/api/activity/delete",{
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
        request.post("/api/activity/deleteBatch",{ids}).then(res=>{
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
        request.post("/api/activity/insert",{...this.detail}).then(res=>{
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
        request.post("/api/activity/update",{...this.detail}).then(res=>{
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
    }
  }
</script>

<style scoped>
</style>
