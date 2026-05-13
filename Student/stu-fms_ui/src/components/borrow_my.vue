<template>
  <div>
    <!--表格-->
    <el-table
      :data="tableData"
      v-loading="tableLoading"
      height="370">
      <el-table-column prop="stuId" label="被借阅学生学号" >
      </el-table-column>
      <el-table-column prop="archiveName" label="档案名称">
      </el-table-column>
      <el-table-column prop="borrTime" label="借阅开始时间">
      </el-table-column>
      <el-table-column prop="borrStatus" label="借阅状态">
        <template slot-scope="scope">
          <span v-if="scope.row.borrStatus===1" style="color: #E6A23C">借阅中</span>
          <span v-if="scope.row.borrStatus===0" style="color:#67C23A">已归还</span>
          <span v-if="scope.row.borrStatus===-1" style="color: #F56C6C">超时未归还</span>
        </template>
      </el-table-column>
      <el-table-column
        fixed="right"
        label="操作"
        width="160">
        <template slot-scope="scope">
          <el-button v-if="scope.row.borrStatus===1" size="mini" type="success" @click="commit(scope.row)">申请归还</el-button>
          <el-button v-if="scope.row.borrStatus===-1" size="mini" type="danger" @click="commit(scope.row)">立即归还</el-button>
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
</template>

<script>
  import request from '../utils/request'

  export default {
    name: 'borrow_my',
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
        var uid=this.$store.state.user.id;
        if(!uid) return;
        this.tableLoading=true;
        request.get("/api/physical/archives/record/getOne",{
          params:{
            usernameId:uid
          }
        }).then(res=>{
          this.tableLoading=false;
          this.total=res.data.data.length;
          this.tableData=res.data.data;
        });
      },
    }

  }
</script>

<style scoped>

</style>
