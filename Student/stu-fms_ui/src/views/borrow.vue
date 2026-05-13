<template>
<!--  档案借阅-->
  <div style="padding:20px;">
    <el-tabs v-model="activeName">
      <el-tab-pane label="可借阅档案" name="可借阅档案">
        <BorrowAllow></BorrowAllow>
      </el-tab-pane>
      <el-tab-pane label="我的借阅记录" name="我的借阅记录">
        <BorrowMy></BorrowMy>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script>
  import BorrowAllow from '@/components/borrow_allow'
  import BorrowMy from '../components/borrow_my'
  export default {
    name: 'borrow',
    components:{
      BorrowAllow,BorrowMy
    },
    data(){
      return {
        activeName:"可借阅档案"
      }
    }
  }
</script>

<style scoped>

</style>
