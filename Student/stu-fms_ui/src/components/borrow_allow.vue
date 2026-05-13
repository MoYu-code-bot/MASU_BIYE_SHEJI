<template>
    <div>
      <!--表格-->
      <el-table
        :data="tableData"
        v-loading="tableLoading"
        height="370">
        <el-table-column prop="stuId" label="学号" >
        </el-table-column>
        <el-table-column prop="archiveName" label="档案名称">
        </el-table-column>
        <el-table-column prop="position" label="存放位置" >
        </el-table-column>
        <el-table-column prop="archiveDesc" label="档案说明" >
          <template slot-scope="scope">
            <el-link type="primary" @click="openDesc(scope.row.archiveDesc)">{{subStr(scope.row.archiveDesc,10)}}</el-link>
          </template>
        </el-table-column>
        <el-table-column
          fixed="right"
          label="操作"
          width="250">
          <template slot-scope="scope">
            <el-button size="mini" type="success" @click="commit(scope.row)">申请借阅</el-button>
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
      <el-dialog
        title="体检结果说明"
        @close="closeDesc"
        :visible.sync="showDesc">
        <span>{{desc}}</span>
      </el-dialog>
    </div>
</template>

<script>
  import request from '../utils/request'

  export default {
    name: 'borrow_allow',
    data(){
      return {
        showMutipleImg:false,
        imgUrls:[], //多图显示的src
        tableData: [{
          stuId:45451
        }],
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
        showDesc:false,
        desc:'暂无描述'
      }
    },
    mounted () {
      this.search();
    },
    methods:{
      closeDesc(){
        this.showDesc=false;
        this.desc="暂无描述";
      },
      openDesc(desc){
        this.showDesc=true;
        this.desc=desc;
      },
      subStr(str,num){
        if(str&&str.length>num){
          str=str.substring(0,num)+"...";
        }
        return  str;
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
        request.get("/api/physical/archives/borrow/list",{
          params:{
            page:this.currentPage,
            size:this.pageSize
          }
        }).then(res=>{
          this.tableLoading=false;
          this.total=res.data.data.total;
          this.tableData=res.data.data.records;
        });
      },
    }

  }
</script>

<style scoped>

</style>
