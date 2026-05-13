<template>
  <div>
    <!-- 表格 -->
    <el-table :data="table1Data" style="width: 100%" @selection-change="getRows">
      <el-table-column type="selection" width="55">
      </el-table-column>

      <el-table-column prop="id" label="档案ID" width="100">
      </el-table-column>
      <el-table-column prop="stuId" label="学号" width="150">
      </el-table-column>
      <el-table-column prop="archiveName" label="档案名称" width="150">
      </el-table-column>
      <el-table-column label="档案图片" width="190">
        <template slot-scope="scope">
          <el-image
            style="width: 100px; height: 100px"
            :src="scope.row.img"
            :fit="fit"></el-image>
        </template>
      </el-table-column>
      <el-table-column prop="position" label="存放位置" width="130">
      </el-table-column>
      <el-table-column label="档案入库时间" width="190">
        <template slot-scope="scope">
          <i class="el-icon-time"></i>
          <span style="margin-left: 10px">{{ scope.row.inTime }}</span>
        </template>
      </el-table-column>
      <el-table-column prop="operatorId" label="操作人员">
      </el-table-column>
      <el-table-column align="right" label="状态">
        <template slot-scope="scope">
          <el-tag type="success" size="small" v-if="scope.row.auditStatusCode =='4'">已入库</el-tag>
          <el-tag type="danger" size="small" v-if="scope.row.auditStatusCode =='5'">已被驳回</el-tag>
          <el-tag type="info" size="small" v-if="scope.row.auditStatusCode =='6'">已出库</el-tag>
        </template>
      </el-table-column>
      <el-table-column align="right" label="操作">
        <template slot-scope="scope">
          <el-button size="mini" type="info" icon="el-icon-upload2"
                     v-if="scope.row.auditStatusCode!='4'" disabled>出库
          </el-button>
          <el-button size="mini" type="info" icon="el-icon-upload2" @click="outStock(scope.row)"
                     v-if="scope.row.auditStatusCode=='4'">出库
          </el-button>
        </template>
      </el-table-column>
      <el-table-column align="right" label="">
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
                   :current-page.sync="this.currentPage" :page-size="this.pageSize" layout="prev, pager, next, jumper"
                   :total="pageNum">
    </el-pagination>

  </div>
</template>

<script>

  import request from '@/utils/request';

  export default {
    name: 'archiver-submit',
    data() {
      return {
        //当前页
        currentPage: 0,
        //每页数量
        pageSize: 3,
        //总页数
        pageNum: 100,
        //表格数据
        table1Data: [],
      }
    },
    created() {
      this.getList(this.currentPage, this.pageSize);
    },
    methods: {
      //获取档单信息列表
      //默认第0页，每页5条数据
      getList: function (page, size) {

        request.get('/api//physical/archives/list', {
            params: {
              page: page,
              size: size
            }
          }
        ).then(res => {
          //获取表格数据
          var data = res.data.data.records;
          //渲染表格数据
          this.table1Data = data;
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
        request.post("/api/physical/archives/delete", data).then(res => {
          if (res.data.msg == "success") {
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
        this.getList(this.currentPage, this.pageSize);
      },
      //出库
      outStock(data) {
        var id = data.id;
        console.log("id",id);
        request.get("/api/physical/archives/outStock",{
          params:{
            id
          }
        }).then(res => {
          if (res.data.msg == "success") {
            this.$message({
              message: '出库成功!',
              type: 'success',
              center: true
            })
          } else {
            this.$message({
              message: '出库失败!',
              type: 'warning',
              center: true
            })
          }
          this.getList(this.currentPage,this.pageSize);
        });


      }
    },

  }
</script>

<style>
</style>
