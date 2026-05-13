<template>
	<!-- 角色管理 -->
	<div id="roleManager" style="padding:10px;">
		<div style="padding:10px 0px;padding-left: 20px;">
      <el-button type="primary" @click="openDetail({})" :loading="insertLoading">
        <i class="el-icon-plus"></i>新增</el-button>
      <span style="width: 10px;height: 2px;display: inline-block"></span>
		</div>
		<!-- 表格部分 -->
		<div id="table">
			<el-table
        :data="tableData"
        v-loading="tableLoading"
        @selection-change="selectionChange"
        height="370"
        border
        style="width: 100%">
        <el-table-column fixed prop="roleName" label="角色名">
        </el-table-column>
				<el-table-column fixed prop="roleDesc" label="描述">
				</el-table-column>
				<el-table-column fixed prop="roleMark" label="标识">
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
		</div>
		<!-- 用来添加权限 -->
<!--		<el-dialog title="添加权限" :visible.sync="showAppendPerm">-->
<!--			&lt;!&ndash; 树形控件 &ndash;&gt;-->
<!--			<el-tree show-checkbox :data="treeData" :props="defaultProps" @check-change="handleNodeCheckChange"></el-tree>-->
<!--			<span slot="footer" class="dialog-footer">-->
<!--				<el-button @click="dialogVisible = false">取 消</el-button>-->
<!--				<el-button type="primary" @click="dialogVisible = false">添加</el-button>-->
<!--			</span>-->
<!--		</el-dialog>-->
    <el-dialog
      title="角色信息"
      :visible.sync="showDetail"
      @close="closeDetail"
      width="35%">
<!--      表单区域-->
      <el-form  :model="detail" class="demo-form-inline">
        <el-form-item label="角色名" required>
          <el-input v-model="detail.roleName" placeholder="请输入角色名"></el-input>
        </el-form-item>
        <el-form-item label="标识" required>
          <el-input v-model="detail.roleMark" placeholder="请输入标识"></el-input>
        </el-form-item>
        <el-tree
          node-key="id"
          ref="tree"
          show-checkbox
          :data="treeData"
          :props="defaultProps"></el-tree>
        <el-form-item label="角色描述">
          <el-input type="textarea" v-model="detail.roleDesc" :rows="4" placeholder="请输入角色描述"></el-input>
        </el-form-item>
      </el-form>
      <span slot="footer" class="dialog-footer">
          <el-button @click="showCreateRole = false">取消</el-button>
          <el-button v-if="detail.id" type="primary" :loading="updateLoading" @click="updateRow">保存</el-button>
          <el-button v-if="!detail.id" type="primary" :loading="insertLoading" @click="insertRow">新增</el-button>
        </span>
    </el-dialog>
	</div>
</template>

<script>
import request from '@/utils/request'
import { Notification } from 'element-ui'
export default {
  name: 'roleManager',
  data () {
    return {
      showAppendPerm: false,
      showCreateRole:false,
      activeName: 1,
      formRole: null,
      treeData: [],
      defaultProps: {
        children: 'children',
        label: 'label'
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
    //加载所有的权限数据
    this.requestPermissionData();
  },
  methods: {
    search () {
      this.tableLoading=true;
      request.get("/api/role/list",{
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
    //请求权限数据
    requestPermissionData(){
      //请求权限的数据，渲染权限树
      request.get("/api/permission/all").then(res=>{
        console.log(res);
        var treeData=[];
        var data=res.data.data;
        for(var i=0;i<data.length;i++){
          var k=-1;//需要添加到的序号，-1表示需要新建
          for(var j=0;j<treeData.length;j++){
            if(treeData[j].label==data[i].type){
              k=j;
              break;
            }
          }
          if(k==-1){
            k=treeData.length;
            treeData[k]={label:data[i].type,children:[]}
          }
          treeData[k].children.push({
            label:data[i].title,
            data:data[i],
            id:data[i].id
          });
        }
        this.treeData=treeData;
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
    //打开详情弹框
    openDetail(row){
      this.showDetail=true;
      this.detail={...row};
      //设置权限
      var permissions=row.permissions;
      if(permissions){
        var checkedIds=[];
        for(var i=0;i<permissions.length;i++){
          checkedIds.push(permissions[i].id);
        }
        var self=this;
        setTimeout(function(){
          self.$refs.tree.setCheckedKeys(checkedIds);
        },200);
      }
    },
    //关闭详情弹框
    closeDetail(){
      this.showDetail=false;
      this.detail={};
      this.$refs.tree.setCheckedKeys([]);
    },
    deleteRow(row){
      this.tableLoading=true;
      request.post("/api/role/delete",{
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
      request.post("/api/role/deleteBatch",{ids}).then(res=>{
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
      //获取选中的权限
      var checkedNods=this.$refs.tree.getCheckedNodes();
      var permissions=[];
      for(var i=0;i<checkedNods.length;i++){
        if(checkedNods[i].data){
          permissions.push(checkedNods[i].data);
        }
      }
      this.detail.permissions=permissions;
      request.post("/api/role/insert",{...this.detail}).then(res=>{
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
      //获取选中的权限
      var checkedNods=this.$refs.tree.getCheckedNodes();
      var permissions=[];
      for(var i=0;i<checkedNods.length;i++){
        if(checkedNods[i].data){
          permissions.push(checkedNods[i].data);
        }
      }
      this.detail.permissions=permissions;
      request.post("/api/role/update",{...this.detail}).then(res=>{
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

<style>
	#table {
		padding: 0px 20px;
	}

	.el-tag {
		margin-right: 10px;
	}
</style>
