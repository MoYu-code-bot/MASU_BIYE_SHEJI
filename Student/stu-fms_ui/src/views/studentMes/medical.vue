<template>
  <div style="padding:20px">
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
    <div class="medicalMes">
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
          width="180">
          <template slot-scope="scope">
            <el-link type="primary" @click="linkToStuInfo(scope.row)">{{scope.row.stuId}}</el-link>
          </template>
        </el-table-column>
        <el-table-column
          prop="medicalTime"
          label="体检时间" width="180">
        </el-table-column>
        <el-table-column
          prop="unit"
          label="体检单位" width="180">
        </el-table-column>
        <el-table-column
          prop="medicalDesc"
          label="体检结果说明"
          width="320">
          <template slot-scope="scope">
            <el-link type="primary" @click="openDesc(scope.row.medicalDesc)">{{subStr(scope.row.medicalDesc,10)}}</el-link>
          </template>
        </el-table-column>
        <el-table-column
          prop="img"
          label="体检结果图片"
          width="150"
          height="300">
          <template slot-scope="scope">
            <el-link type="primary" @click="openMutipleImg(scope.row.img)">查看结果</el-link>
          </template>
        </el-table-column>
        <el-table-column
          prop="auditStatusCode"
          width="150"
          label="档案状态">
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
    <el-dialog @close="closeDetail" title="体检信息" :visible.sync="showDetail">
      <el-form :model="detail">
        <el-form-item label="学号:" required :label-width="formLabelWidth" >
          <el-input
            maxlength="20"
            show-word-limit
            v-model="detail.stuId" autocomplete="off"></el-input>
        </el-form-item>
        <el-form-item label="体检单位：" required :label-width="formLabelWidth">
          <el-input v-model="detail.unit" autocomplete="off"></el-input>
        </el-form-item>
        <el-form-item label="体检结果说明：" required :label-width="formLabelWidth">
          <el-input type="textarea"
                    :rows="6"
                    placeholder="请输入体检结果说明"
                    maxlength="500"
                    show-word-limit
                    v-model="detail.medicalDesc" autocomplete="off"></el-input>
        </el-form-item>
        <el-form-item label="体检时间" required :label-width="formLabelWidth">
          <el-date-picker
            v-model="detail.medicalTime"
            type="date"
            placeholder="选择日期"
            style="width: 207.27px">
          </el-date-picker>
        </el-form-item>
        <el-form-item label="参与/获得证明" required :label-width="formLabelWidth" >
          <el-upload
            ref="upload"
            :file-list="imgsList"
            action="/api/upload/img"
            :on-success="uploadSuccess"
            list-type="picture-card"
          >
            <i slot="default" class="el-icon-plus"></i>
            <div slot="file" slot-scope="{file}" id="img">
              <img
                class="el-upload-list__item-thumbnail"
                :src="file.url" alt="">
              <span class="el-upload-list__item-actions">
                <span class="el-upload-list__item-preview"
                      @click="handlePictureCardPreview(file)">
                  <i class="el-icon-zoom-in"></i>
                </span>
              <span
                v-if="!disabled"
                class="el-upload-list__item-delete"
                @click="handleRemove(file.url)">
                <i class="el-icon-delete"></i>
              </span>
            </span>
            </div>
          </el-upload>
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button @click="dialogFormVisible = false">取 消</el-button>
        <el-button v-if="detail.id" type="primary" :loading="updateLoading" @click="updateRow">保存</el-button>
        <el-button v-if="!detail.id" type="primary" :loading="insertLoading" @click="insertRow">新增</el-button>
        <!--        <el-button type="primary" @click="submit('ruleForm')">提交</el-button>-->
      </div>
    </el-dialog>
    <el-dialog
      title="体检结果说明"
      @close="closeDesc"
      :visible.sync="showDesc">
      <span>{{desc}}</span>
    </el-dialog>
    <el-dialog title="证明材料"
               :visible.sync="showMutipleImg"
               @close="closeMutipleImg">
      <span v-if="imgUrls.length==0">未上传体检结果图片</span>
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
    name: 'medical',
    data () {
      return {
        formLabelWidth: '130px',
        imgsList:[],
        showMutipleImg:false,
        imgUrls:[], //多图显示的src
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
        showDesc:false,
        desc:"",
      }
    },
    mounted () {
      this.search();
    },
    methods: {
      openDesc(desc){
        this.showDesc=true;
        this.desc=desc;
      },
      closeDesc(){
        this.showDesc=false;
        this.desc="暂无描述";
      },
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
        request.get("/api/medical/search",{
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
        request.post("/api/medical/delete",{
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
        request.post("/api/medical/deleteBatch",{ids}).then(res=>{
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
        request.post("/api/medical/insert",{...this.detail}).then(res=>{
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
        request.post("/api/medical/update",{...this.detail}).then(res=>{
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

</style>
