<template>
  <div>
    <!-- 选项卡 -->
    <el-tabs type="border-card">
      <!-- 档案信息表 -->
      <el-tab-pane>
        <span slot="label"><i class="el-icon-date"></i>档案信息</span>
        <archives-info></archives-info>
      </el-tab-pane>
      <!-- 档案审核 -->
      <el-tab-pane>
        <span slot="label"><i class="el-icon-date"></i>档案审核</span>
         <archives-check></archives-check>
      </el-tab-pane>
      <!--归档建目-->
      <el-tab-pane>
        <span slot="label"><i class="el-icon-date"></i> 归档建目</span>
        <archives-title></archives-title>
      </el-tab-pane>
      <!--提交入库-->
      <el-tab-pane>
        <span slot="label"><i class="el-icon-date"></i> 提交入库</span>
          <archives-submit></archives-submit>
      </el-tab-pane>
    </el-tabs>

  </div>

</template>

<script>
import archivesInfo from '@/components/archives_info'
import archivesCheck from '@/components/archives_check'
import archivesTitle from '@/components/archives_title'
import archivesSubmit from '@/components/archives_submit'
export default {
  components: { archivesInfo, archivesCheck, archivesTitle, archivesSubmit },
  data () {
    I
    return {

    }
  },
  methods: {

  }

}
</script>

<style>
</style>
