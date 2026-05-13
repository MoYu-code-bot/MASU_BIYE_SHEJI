<template>
  <div>
    <!-- 档案信息表 -->
    <el-button type="primary" size="medium" icon="el-icon-plus" @click="add">新增</el-button>
    <el-button type="danger" size="medium" icon="el-icon-delete" @click="batchDelete">批量删除</el-button>
    <el-button type="infor" size="medium" icon="el-icon-s-data" @click="testData">数据统计</el-button>
    <!-- 搜索框 -->
    <el-form :inline="true" style="display: inline;float: right;" class="demo-form-inline;">
      <el-form-item>
        <el-input placeholder="请输入用户名...." v-model="searchData" :inline="true">
          <el-button slot="append" icon="el-icon-search" @click="search"></el-button>
        </el-input>
      </el-form-item>
    </el-form>
    <!-- 表格 -->
    <el-table :data="table1Data" style="width: 100%" @selection-change="getRows">
      <el-table-column type="selection" width="55">
      </el-table-column>
      <el-table-column prop="id" v-if="false">
      </el-table-column>
      <el-table-column prop="userId" label="用户ID" width="180">
      </el-table-column>
      <el-table-column prop="username" label="用户名" width="150">
      </el-table-column>
      <el-table-column label="档案创建时间" width="190">
        <template slot-scope="scope">
          <i class="el-icon-time"></i>
          <span style="margin-left: 10px">{{ scope.row.created }}</span>
        </template>
      </el-table-column>
      <el-table-column prop="borrCount" label="借阅次数">
      </el-table-column>
      <el-table-column prop="cutOffTime" label="截止时间">
        <template slot-scope="scope">
          <i class="el-icon-time"></i>
          <span style="margin-left: 10px">{{ scope.row.cutOffTime }}</span>
        </template>
      </el-table-column>
      <el-table-column align="right" size="200">
        <!-- 编辑、删除按钮 -->
        <template slot-scope="scope">
          <el-button size="mini" type="infor" icon="el-icon-edit" @click="edit(scope.row)">修改</el-button>
          <el-button size="mini" type="danger" icon="el-icon-delete" @click="del(scope.row)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- 对话框 -->
    <el-dialog :title="dialogTitle" :visible.sync="dialogFormVisible" width="34%">
      <el-form :model="form" class="demo-form-inline">
        <el-form-item label="用户ID" :label-width="formLabelWidth">
          <el-input v-model="form.userId" autocomplete="off" style="width: 218px;"></el-input>
        </el-form-item>
        <el-form-item label="用户名" :label-width="formLabelWidth">
          <el-input v-model="form.username" autocomplete="off" style="width: 218px;"></el-input>
        </el-form-item>
        <el-form-item label="创建时间" :label-width="formLabelWidth">
          <el-date-picker v-model="form.created" type="date" value-format="yyyy-MM-dd" placeholder="选择日期">
          </el-date-picker>
        </el-form-item>
        <el-form-item label="借阅次数" :label-width="formLabelWidth">
          <el-input v-model="form.borrCount" autocomplete="off" style="width: 218px;"></el-input>
        </el-form-item>
        <el-form-item label="截止时间" :label-width="formLabelWidth">
          <el-date-picker v-model="form.cutOffTime" type="date" value-format="yyyy-MM-dd" placeholder="选择日期">
          </el-date-picker>
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button @click="cancelDialog">取 消</el-button>
        <el-button type="primary" @click="submit">确 定</el-button>
      </div>
    </el-dialog>
    <!-- 分页 -->
    <el-pagination align="right" @current-change="handleCurrentChange"
                   :current-page.sync="this.currentPage" :page-size="this.pageSize" layout="prev, pager, next, jumper"
                   :total="pageNum">
    </el-pagination>
  </div>
</template>

<script>

  //引用api
  import request from '@/utils/request';

  export default {
    name: 'archives-info',
    data() {
      return {
        //当前页
        currentPage: 0,
        //每页数量
        pageSize: 5,
        //总页数
        pageNum: 100,
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
        form: {
          id: '',
          userId: '',
          created: '',
          borrCount: '',
          cutOffTime: ''
        },
        // 档案信息表
        table1Data: [],

      }
    },

    //创建时调用的函数
    created() {

      this.getList(this.currentPage, this.pageSize);

    },

    methods: {
      //获取档单信息列表
      //默认第0页，每页5条数据
      getList: function (page, size) {

        request.get('/api/archive/list', {
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

      // 新增
      add() {
        this.from = {};
        this.dialogType = 'add';
        // 打开对话框
        this.dialogTitle = '新增';
        this.dialogFormVisible = true;

      }
      ,
      // 编辑
      edit(data) {
        this.dialogType = 'edit'
        // 打开对话框
        this.dialogTitle = '编辑'
        this.dialogFormVisible = true
        //console.log('edit', data)
        // 渲染数据到编辑表单里
        this.form.id = data.id;
        this.form.userId = data.userId;
        this.form.username = data.username;
        this.form.borrCount = data.borrCount;
        this.form.created = data.created;
        this.form.cutOffTime = data.cutOffTime;
      }
      ,
      // 提交数据
      submit() {
        //新增
        if (this.dialogType == 'add') {

          //获取表格信息
          var form = this.form;
          console.log("新增", form);

          request.post('/api/archive/insert', form).then(res => {

            if (res.data.status == 0) {
              this.$message({
                type: 'success',
                message: '新增成功!',
                center: true
              });
            } else {
              this.$message({
                type: 'warning',
                message: '新增失败!',
                center: true
              });

            }

          }).catch(error => {
            console.log("请求api错误" + "/api/archives/insert", error);
          });

          // 关闭对话框
          this.dialogFormVisible = false;


          //重新渲染数据
          this.getList(this.currentPage, this.pageSize);

          //清空表格
          this.form = {};

        }
        //编辑
        if (this.dialogType == 'edit') {
          var form = this.form;
          console.log(this.form);
          request.post('/api/archive/update', form).then(res => {
            if (res.data.status == 0) {
              this.$message({
                type: 'success',
                message: '修改成功!',
                center: true
              });
            } else {
              this.$message({
                type: 'warning',
                message: '修改失败!',
                center: true
              });

            }

            console.log("res=", res)
          }).catch(error => {
            console.log("请求api错误" + "/api/archives/update");
          });

          // 发送修改请求
          console.log('发送修改请求')
          // 关闭对话框
          this.dialogFormVisible = false;
          //重新渲染数据
          this.getList(this.currentPage, this.pageSize);
          //清空form
          this.form = {};
        }
      }
      ,
      // 搜索
      search() {
        var data = this.searchData;
        console.log('searchData', data);

        request.get("/api/archive/search", {
          params: {
            page: 0,
            size: this.pageSize,
            searchValue: data
          }
        }).then(res => {

          //获取表格数据
          var data = res.data.data.records;
          //渲染表格数据
          this.table1Data = data;
          //获取总页数
          this.pageNum = res.data.data.total;

        }).catch(error => {
          console.log(error);
        })

      }
      ,
      // 删除
      del(data) {

        //构建删除对象
        var deleteData = {
          id: '',
          userId: '',
          username: '',
          borrCount: '',
          created: '',
          cutOffTime: '',
        };

        deleteData.id = data.id;
        deleteData.userId = data.userId;
        deleteData.username = data.username;
        deleteData.borrCount = data.borrCount;
        deleteData.created = data.created;
        deleteData.cutOffTime = data.cutOffTime;

        console.log('删除行的数据', deleteData);

        this.$confirm('是否确定删除该数据, 是否继续?', '提示', {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'warning'
        }).then(() => {

            request.post('/api/archive/delete', deleteData).then(res => {

              console.log("删除", res);

              if (res.data.status == 0) {
                this.$message({
                  type: 'success',
                  message: '删除成功!',
                  center: true
                });
              }
              else {
                this.$message({
                  type: 'warning',
                  message: '删除失败!',
                  center: true
                });
              }

              //重新渲染数据
              this.getList(this.currentPage, this.pageSize);


            }).catch(error => {
              console.log("api请求失败", error);
            });


          }
        ).catch(() => {
          this.$message({
            type: 'info',
            message: '已取消删除',
            center: true
          });
        });

      }
      ,
      // 获取多选框的数据
      getRows(rows) {
        this.selectIds = [];
        for (var i = 0; i < rows.length; i++) {
          this.selectIds.push(rows[i].id);
        }
        // console.log("ids",this.selectIds);
      },

      // 批量删除
      batchDelete() {
        // 获取被选的多选框的值
        var ids = this.selectIds;
        var idsObj = {ids: []};
        idsObj.ids = ids;
        if (ids.length <= 0) {
          this.$message({
            showClose: true,
            message: '请至少选择一点数据删除',
            type: 'warning',
            center: true
          })
          return;
        }
        console.log('需要删除的ids', idsObj);

        request.post('/api/archive/deleteBatch', idsObj).then(res => {
          console.log("批量删除", res);

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

          //重新渲染数据
          this.getList(this.currentPage, this.pageSize);

        }).catch(error => {

        });

      }
      ,
      //获取当前页数
      handleCurrentChange(page) {

        this.getList(page, this.pageSize);
        //设置当前页数
        this.currentPage = page;

      },

      //
      cancelDialog() {
        this.dialogFormVisible = false;
        this.form = {};
      }


    }

  }
</script>

<style>
</style>
