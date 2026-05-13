<template>
  <div>
    <el-table :data="tableData" stripe style="width: 100%">
      <el-table-column prop="title" label="标题" width="180">
      </el-table-column>
      <el-table-column prop="number" label="数量" width="180">
      </el-table-column>
      <el-table-column prop="updateTime" label="最新更新时间" width="180">
      </el-table-column>
    </el-table>
    <!-- 分页 -->
    <el-pagination align="right" @size-change="handleSizeChange" @current-change="handleCurrentChange"
      :current-page.sync="currentPage3" :page-size="8" layout="prev, pager, next, jumper" :total="1000">
    </el-pagination>
  </div>
</template>

<script>
export default {
  name: 'archives-title',
  data () {
    return {
      currentPage3: 5,
      tableData: [
        {
          title: '管理类',
          number: '15',
          updateTime: '2017-12-4'
        },
        {
          title: '社会类',
          number: '13',
          updateTime: '2017-12-4'
        },
        {
          title: '文学类',
          number: '15',
          updateTime: '2017-12-4'
        }
      ]

    }
  },
  methods: {
    // 分页
    handleSizeChange (val) {
      console.log(`每页 ${val} 条`)
    },
    handleCurrentChange (val) {
      console.log(`当前页: ${val}`)
    }
  }
}
</script>

<style>
</style>
