<template>
  <el-tabs type="border-card">
    <el-tab-pane>
      <span slot="label"><i class="el-icon-date"></i>档案借阅记录</span>
      <archives-borrow-info></archives-borrow-info>
    </el-tab-pane>
    <el-tab-pane>
      <span slot="label"><i class="el-icon-date"></i>档案借阅状态</span>
      <archive-borrow-status></archive-borrow-status>
    </el-tab-pane>
  </el-tabs>
</template>

<script>
import ArchivesBorrowInfo from '@/components/archives_borrow_info'
import ArchiveBorrowStatus from '@/components/archive_borrow_status'
export default {
  components: {
    ArchivesBorrowInfo,
    ArchiveBorrowStatus
  },
  data () {
    return {

    }
  }
}
</script>

<style>
</style>
