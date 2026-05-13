<template>
  <div>
    <!-- 搜索框 -->
    <el-form :inline="true" style="display: inline;float: right;" class="demo-form-inline;">
      <el-form-item>
        <el-input placeholder="请输入借阅者...." v-model="searchData" :inline="true">
          <el-button slot="append" icon="el-icon-search" @click="search"></el-button>
        </el-input>
      </el-form-item>
    </el-form>

    <el-table height="370" :data="tableData" style="width: 100%" @selection-change="getRows">
      <el-table-column type="selection" width="55">
      </el-table-column>
      <el-table-column prop="archiveId" label="档案ID">
      </el-table-column>
      <el-table-column prop="archiveName" label="档案名称">
      </el-table-column>
      <el-table-column prop="stuName" label="档案拥有者">
      </el-table-column>
      <el-table-column prop="username" label="借阅者">
      </el-table-column>
      <el-table-column label="档案借阅时间" width="190">
        <template slot-scope="scope">
          <i class="el-icon-time"></i>
          <span style="margin-left: 10px">{{ scope.row.borrTime }}</span>
        </template>
      </el-table-column>
      <el-table-column label="档案借阅状态" width="190">
        <template slot-scope="scope">
          <el-tag type="success" size="mini" v-if="scope.row.borrStatus == '0'">借阅中</el-tag>
          <el-tag type="danger" size="mini" v-if="scope.row.borrStatus == '1'">已归还</el-tag>
          <el-tag type="info" size="mini" v-if="scope.row.borrStatus == '-1' ">借阅超时</el-tag>
        </template>
      </el-table-column>
      <el-table-column label="操作">
        <template slot-scope="scope">
          <el-popconfirm
            confirmButtonText='确定'
            cancelButtonText='取消'
            icon="el-icon-info"
            iconColor="red"
            title="是否确定删除该数据吗？"
            @onConfirm="delet(scope.row)"
          >
            <el-button slot="reference" size="mini" type="danger">删除</el-button>
          </el-popconfirm>
        </template>
      </el-table-column>
    </el-table>
    <!-- 分页 -->
    <el-pagination align="right" @current-change="handleCurrentChange"
                   :current-page.sync="currentPage3" :page-size="pageSize" layout="prev, pager, next, jumper"
                   :total="pageNum">
    </el-pagination>

  </div>
</template>

<script>

  import request from '@/utils/request';

  export default {
    name: 'archive-borrow-info',
    data() {
      return {
        //当前页
        currentPage: 0,
        //每页数量
        pageSize: 5,
        //总页数
        pageNum: 100,
        searchData: '',
        // 档案信息表
        tableData: [],
        //当前页
        currentPage3: 0,
      }
    },
    created() {

      this.getList(this.currentPage, this.pageSize);
    },
    methods: {
      //获取档单信息列表
      //默认第0页，每页5条数据
      getList: function (page, size) {

        request.get('/api/physical/archives/record/list', {
            params: {
              page: page,
              size: size
            }
          }
        ).then(res => {
          //获取表格数据
          var data = res.data.data.records;
          //渲染表格数据
          this.tableData = data;
          //获取总页数
          this.pageNum = res.data.data.total;
          console.log(res);
          console.log("总页数", this.pageNum);

        }).catch(error => {
          console.log("api请求失败", error);
        })

      },
      //获取当前页数
      handleCurrentChange(page) {

        this.getList(page, this.pageSize);
        //设置当前页数
        this.currentPage = page;

      },
      //删除
      delet(data) {
        console.log("删除", data);

        request.post("/api/physical/archives/record/delete", data).then(res => {
          if (res.data.status == 0) {
            this.$message({
              message: '删除成功!',
              type: 'success',
              center: true
            })
          } else {
            this.$message({
              message: '删除失败!',
              type: 'warning',
              center: true
            })
          }
        }).catch(error => {

        });
        //重新渲染数据
        this.getList(this.currentPage,this.pageSize);
      },


      //搜索
      search(){
        console.log(this.searchData);
        request.get("/api/physical/archives/record/search",{
          params:{
            page:this.currentPage,
            size:this.pageSize,
            searchValue:this.searchData
          }
        }).then( res =>{
          var data = res.data.data.records;
          this.tableData =data;
          //获取总页数
          this.pageNum = res.data.data.total;
        }).catch( error =>{
          console.log(error);
        })
      }

    },
  }
</script>

<style>
</style>
