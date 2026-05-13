<template>
    <div class="course" style="padding:20px;">
<!--      查询部分-->
      <div id="query" style="padding-bottom: 10px">
        <el-button type="primary" @click="openCourseDetail({})" :loading="insertLoading">
          <i class="el-icon-plus"></i>新增</el-button>
        <span style="width: 10px;height: 2px;display: inline-block"></span>
        <el-popconfirm
          title="确定删除这些用户吗？"
          @onConfirm="deleteBatch">
          <el-button slot="reference" type="danger" @click="" :loading="deleteBatchLoading">
            <i class="el-icon-delete"></i>批量删除</el-button>
        </el-popconfirm>
        <div style="width:30%;float: right">
          <el-input @keyup.enter.native="search" placeholder="请输入内容" v-model="searchValue" class="input-with-select">
            <el-button @click="search" slot="append" icon="el-icon-search" :loading="searchLoading"></el-button>
          </el-input>
        </div>
      </div>
      <div class="courseGrade">
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
            prop="courseName"
            label="课程名称"
            width="280">
          </el-table-column>
          <el-table-column
            prop="score"
            label="课程成绩"
            width="240">
            <template slot-scope="scope">
              <el-link type="primary" @click="lookCourse(scope.row)">
                {{computeScore(scope.row)}}
              </el-link>
            </template>
          </el-table-column>
          <el-table-column
            prop="courseLevel"
            label="课程等级"
            width="240">
            <template slot-scope="scope">
              <span v-if="scope.row.courseLevel=='优'" style="color: #85ce61" >优</span>
              <span v-if="scope.row.courseLevel=='良'" style="color: #409eff" >良</span>
              <span v-if="scope.row.courseLevel=='中'" style="color: #e6a23c" >中</span>
              <span v-if="scope.row.courseLevel=='差'" style="color: #f56c6c" >差</span>
            </template>
          </el-table-column>
          <el-table-column
            prop="courseRank"
            label="课程排名"
            width="240">
          </el-table-column>
          <el-table-column
            prop="auditStatusCode"
            label="状态"
            width="240">
            <template slot-scope="scope">
              <span>{{$dict.getValue(scope,'audit_status',scope.row.auditStatusCode)}}</span>
            </template>
          </el-table-column>
          <el-table-column
            prop="teacher"
            label="任课教师"
            >
          </el-table-column>
          <el-table-column
            fixed="right"
            label="操作"
            width="250">
            <template slot-scope="scope">
              <el-button @click="openCourseDetail(scope.row)" type="primary" size="small">修改</el-button>
              <span style="width: 10px;height: 2px;display: inline-block"></span>
              <el-popconfirm
                title="确定删除这个用户吗？"
                v-model="showDeleteConfirm"
                @onConfirm="deleteCourse(scope.row)">
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
        <el-dialog title="成绩详情" :visible.sync="showScoreDialog" width="15%" :before-close="closeDialog">
          平时成绩(<span>{{course.usualRatio*100}}%</span>)：<span style="color: #e6a23c">{{course.usualScore}}</span><br>
          卷面成绩(<span>{{(1-course.usualRatio)*100}}%</span>)：<span style="color: #85ce61">{{course.score}}</span><br>
          成绩等级：<span>{{course.courseLevel}}</span>
        </el-dialog>
<!--        课程详情-->
        <el-dialog title="成绩信息" :visible.sync="showCourseDetail" width="30%" :before-close="closeDialog">
          <el-form :model="course">
            <el-form-item label="学号:" required>
              <el-input v-model="course.uid"  ></el-input>
            </el-form-item>
            <el-form-item label="课程名称:" required>
              <el-input v-model="course.courseName"  ></el-input>
            </el-form-item>
            <el-form-item label="卷面成绩:">
              <el-input type="number" v-model="course.score" ></el-input>
            </el-form-item>
            <el-form-item label="平时成绩:">
              <el-input type="number" v-model="course.usualScore" ></el-input>
            </el-form-item>
            <el-form-item label="平时成绩占比：">
              <el-input type="number" v-model="course.usualRatio" ></el-input>
            </el-form-item>
            <el-form-item label="总成绩：">
              <el-input :disabled="true" type="number" :value="computeScore(course)" readonly></el-input>
            </el-form-item>
            <el-form-item label="成绩等级：">
              <el-select v-model="course.courseLevel" placeholder="请选择">
                <el-option label="优" value="优"></el-option>
                <el-option label="良" value="良"></el-option>
                <el-option label="中" value="中"></el-option>
                <el-option label="差" value="差"></el-option>
              </el-select>
            </el-form-item>
            <el-form-item label="排名：">
              <el-input type="number" v-model="course.courseRank" ></el-input>
            </el-form-item>
            <el-form-item label="任课老师：" required>
              <el-input v-model="course.teacher" ></el-input>
            </el-form-item>
          </el-form>
          <span slot="footer" class="dialog-footer">
            <el-button @click="closeCourseDetail">取 消</el-button>
            <el-button v-if="course.id" type="primary" :loading="updateLoading" @click="updateCourse">保存</el-button>
            <el-button v-if="!course.id" type="primary" :loading="insertLoading" @click="insertCourse">新增</el-button>
          </span>
        </el-dialog>
      </div>
    </div>
</template>

<script>
import request from '../../utils/request'
import { Notification } from 'element-ui'
export default {
  name: 'courseMes',
  data () {
    return {
      labelPosition:"top",
      tableData: [],
      tableLoading:false,
      currentPage:1,
      pageSize:10,
      searchValue:"",
      total:0,
      showDeleteConfirm: false,
      showScoreDialog:false,
      course:{},
      updateLoading:false,
      insertLoading:false,
      showCourseDetail:false,
      deleteBatchLoading:false,
      selectedRows:[]
    }
  },
  mounted () {
    this.search(this.currentPage,this.pageSize);
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
    //搜索
    search (page,size) {
      this.tableLoading=true;
      request.get("/api/course/search",{
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
    handleCurrentChange(curPage){
      this.currentPage=curPage;
      this.search(this.currentPage,this.pageSize);
    },
    handleSizeChange(size){
      this.pageSize=size;
      this.search(this.currentPage,this.pageSize);
    },
    lookCourse(row){
      this.showScoreDialog=true;
      this.course=row;
    },
    updateCourse(){
      this.updateLoading=true;
      request.post("/api/course/update",{...this.course}).then(res=>{
        this.updateLoading=false;
        this.closeCourseDetail();
        Notification({
          type: 'success',
          message: '保存成功'
        })
        //刷新表格
        this.search(this.currentPage,this.pageSize);
      });
    },
    deleteCourse(row){
      this.tableLoading=true;
      request.post("/api/course/delete",{
        id:row.id
      }).then(res=>{
        if(res.data.status==0){
          this.tableLoading=false;
          Notification({
            type: 'success',
            message: '删除成功'
          })
          //刷新表格
          this.search(this.currentPage,this.pageSize);
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
      request.post("/api/course/deleteBatch",{ids}).then(res=>{
        this.deleteBatchLoading=false;
        if(res.data.status==0){
          Notification({
            type: 'success',
            message: '批量删除成功'
          })
          //刷新数据
          this.search(this.currentPage,this.pageSize);
        }
      });
    },
    insertCourse(){
      this.insertLoading=true;
      request.post("/api/course/insert",{...this.course}).then(res=>{
        this.insertLoading=false;
        if(res.data.status==0){
          Notification({
            type: 'success',
            message: '新增成功'
          })
          this.closeCourseDetail()
          //刷新数据
          this.search(this.currentPage,this.pageSize);
        }
      });
    },
    openCourseDetail(row){
      this.showCourseDetail=true;
      this.course={...row};
    },
    closeCourseDetail(){
      this.showCourseDetail=false;
      this.course={};
    },
    selectionChange(val){
      this.selectedRows=val;
    },
    //计算成绩
    computeScore(course){
      var score=course.score*(1-course.usualRatio)+course.usualScore*course.usualRatio;
      score=score.toFixed(1);
      return score;
    }
  }
}
</script>

<style scoped>
</style>
