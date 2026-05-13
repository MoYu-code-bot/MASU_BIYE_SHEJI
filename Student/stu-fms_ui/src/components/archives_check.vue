<template>
  <div>
    <el-button type="danger" size="medium" icon="el-icon-delete" @click="batchDelete">批量删除</el-button>
    <el-button type="success" size="medium" icon="el-icon-finished" @click="batchAudit">批量审核</el-button>
    <!-- 搜索框 -->
    <el-form :inline="true" style="display: inline;float: right;" class="demo-form-inline;">
      <el-form-item>
        <el-input placeholder="请输入学号...." v-model="searchData" :inline="true">
          <el-button slot="append" icon="el-icon-search" @click="search"></el-button>
        </el-input>
      </el-form-item>
    </el-form>
    <el-table height="370" :data="table2Data" style="width: 100%" @selection-change="getRows">
      <el-table-column type="selection" width="55">
      </el-table-column>
      <el-table-column prop="id" label="档案ID">
      </el-table-column>
      <el-table-column prop="stuId" label="学号">
      </el-table-column>
      <el-table-column prop="dictDesc" label="档案类型">
      </el-table-column>
      <el-table-column label="操作">
        <template slot-scope="scope">
          <el-button type="info" size="mini" @click="$router.push('/userManager')">详情...</el-button>
          <el-button type="danger" size="mini" v-if="scope.row.isAudited== '0'" @click="audit(scope.row.id)">审核
          </el-button>
          <el-button type="success" size="mini" v-if="scope.row.isAudited== '1'" disabled>审核已通过</el-button>
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

    name: 'archives-check',
    data() {
      return {
        //点击行数据
        rowData: {},
        //被选择的ids
        selectIds: [],
        //当前页
        currentPage: 1,
        //每页数量

        pageSize: 5,

        pageSize: 10,
        //总页数
        pageNum: 100,
        dialogVisible: false,
        selectIds: [], // 被选的的ids
        searchData: '',
        dialogType: '', // 对话框类型，区分新增还是编辑
        dialogTitle: '',
        value1: '',
        value2: '',
        currentPage3: 0,
        pickerOptions: {
          disabledDate(time) {
            return time.getTime() > Date.now()
          }
        },
        dialogFormVisible: false,
        formLabelWidth: '120px',

        // 档案审核信息表
        table2Data: [],

      }
    },
    created() {
      this.getList(this.currentPage, this.pageSize);
    },

    methods: {
      // 分页
      handleCurrentChange(val) {
        this.getList(val, this.pageSize);
        this.currentPage = val;
        console.log('当前页:', this.currentPage);

      },

      //获取档单信息列表
      //默认第0页，每页5条数据
      getList: function (page, size) {

        request.get('/api/verify/list', {
            params: {
              page: page,
              size: size
            }
          }
        ).then(res => {
          //获取表格数据
          console.log("res", res);
          var data = res.data.data.records;
          //渲染表格数据
          this.table2Data = data;
          //获取总页数
          this.pageNum = res.data.data.total;
          console.log("审核数据", data);
          console.log("总页数", this.pageNum);

        }).catch(error => {
          console.log("api请求失败", error);
        })

      },

      // 批量审核
      batchAudit() {
        // 获取被选的多选框的值
        var ids = this.selectIds
        if (ids.length <= 0) {
          this.$message({
            showClose: true,
            message: '请至少选择一条数据审核',
            type: 'warning',
            center: true
          })
        }
        console.log('需要审核的ids', ids)
      },
      // 搜索
      search() {
        var data = this.searchData;
        console.log('searchData', data);
      },
      // 批量删除
      batchDelete() {
        // 获取被选的多选框的值
        var ids = this.selectIds
        if (ids.length <= 0) {
          this.$message({
            showClose: true,
            message: '请至少选择一点数据删除',
            type: 'warning',
            center: true
          })
          return;
        }
        var deleteData = {};
        deleteData.ids = ids;
        console.log('需要删除的数据', deleteData);

        request.post("/api/verify/deleteBatch", deleteData).then(res => {
          if (res.data.status == 0) {
            this.$message({
              message: '批量删除成功!',
              type: 'success',
              center: true
            })
          } else {
            this.$message({
              message: '批量删除失败!',
              type: 'warning',
              center: true
            })
          }

        }).catch(error => {
          console.log(error);
        })
        //重新渲染数据
        this.getList(1, this.pageSize);
      },

      // 获取多选框的数据
      getRows(rows) {
        this.selectIds = [];
        for (var i = 0; i < rows.length; i++) {
          this.selectIds.push(rows[i].id);
        }
        // console.log("ids",this.selectIds);
      },

      //审核
      audit(id) {
        console.log("审核id", id);
        request.get("/api/verify/audit", {
          params: {
            id
          }
        }).then(res => {
          if (res.data.status == 0) {
            this.$message({
              message: '审核成功!',
              type: 'success',
              center: true
            })
          } else {
            this.$message({
              message: '审核失败!',
              type: 'warning',
              center: true
            })
          }
          //重新渲染数据
          this.getList(this.currentPage, this.pageSize);
        }).catch(error => {
          console.log(error);
        })


      }
    }
  }
</script>

<style>
  .grid-content {
    border-radius: 4px;
    font-size: 18px;
    min-height: 50px;
  }
</style>
