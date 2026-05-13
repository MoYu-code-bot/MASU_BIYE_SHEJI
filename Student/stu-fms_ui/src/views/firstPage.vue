<template>
  <div>
    <div class="user-mes" :style="{width: '100%', height: '110px'}">
      <!--用户信息-->
        <div style="background: #7fdbda;width: 31%;height: 100%">
          <i class="el-icon-user-solid"></i>
          <span>总用户数量：300</span>
        </div>
        <div style="background: #848ccf;width: 31%;height: 100%">
          <i class="el-icon-s-custom"></i>
          <span>学生数：{{this.student}}</span>
        </div>
        <div style="background: #F56C6C;width: 31%;height: 100%">
          <i class="el-icon-s-check"></i>
          <span>教职工数：66</span>
        </div>
    </div>
    <div class="charts">
      <!--档案借阅记录信息-->
      <div id="file-borrow" :style="{width: '600px', height: '400px'}"></div>
      <!--档案信息-->
      <div id="stu-files" :style="{width: '600px', height: '400px'}"></div>
      <!--就业信息-->
      <div id="stu-jobs" :style="{width: '600px', height: '400px'}"></div>
    </div>
  </div>

</template>

<script>
    export default {
        name: "firstPage",
        data(){
          return {
            student:234,
            list:[30.3, 20.5, 56.8, 60.9, 90, 80]
          }
        },
      mounted(){
        this.drawLine();
      },
      methods: {
        drawLine(){
          var self = this;
          // 基于准备好的dom，初始化echarts实例,档案信息
          let files = this.$echarts.init(document.getElementById('stu-files'))
          // 绘制图表
          files.setOption({
            title: {
              text: '档案数据详情',
              textStyle: {
                color:'#777'
              }
            },
            tooltip: {},
            xAxis: {
              data: ["分类1","分类2","分类3","分类4","分类5"]
            },
            yAxis: {},
            series: [{
              name: '数量',
              type: 'bar',
              data: [5, 20, 36, 10, 78]
            }]
          });
          //就业信息
          let jobs = this.$echarts.init(document.getElementById('stu-jobs'));
          jobs.setOption({
            title: {
              text: '就业信息详情',
              textStyle: {
                color:'#777'
              }
            },
            tooltip: {
              trigger: 'axis',
              axisPointer: {
                type: 'cross',
                crossStyle: {
                  color: '#999'
                }
              },
              formatter: function (params) {
                //console.log(params[0]); // 当我们想要自定义提示框内容时，可以先将鼠标悬浮的数据打印出来，然后根据需求提取出来即可
                let firstParams = params[0];
                return '专业名称：'+firstParams.name + '<br>' + firstParams.seriesName + '：' + firstParams.data +' %';
              }
            },
            xAxis: {
              data: ["专业1","专业2","专业3","专业4","专业5","专业6"]
            },
            yAxis: {},
            series: [{
              name: '就业率',
              type: 'bar',
              data: this.list
            }]
          });
          //档案借阅记录信息
          let counts = this.$echarts.init(document.getElementById('file-borrow'));
          counts.setOption({
            title: {
              text: '档案借阅记录',
              textStyle: {
                color:'#777'
              }
            },
            tooltip: {
              trigger: 'item',
              axisPointer: {
                type: 'cross',
                crossStyle: {
                  color: '#999'
                }
              },
              formatter: function (params) {
                //console.log(params)
                let firstParams = params;
                return firstParams.name + ':' + firstParams.data.value +' %';
              }
            },
            series : [
              {
                name: '',
                type: 'pie',    // 设置图表类型为饼图
                radius: '55%',  // 饼图的半径，外半径为可视区尺寸（容器高宽中较小一项）的 55% 长度。
                data:[          // 数据数组，name 为数据项名称，value 为数据项值
                  {value:67.2, name:'入库记录'},
                  {value:56.8, name:'借阅'},
                  {value:10.4, name:'超时借阅'},
                ],
                color: ['#73b9bc','#759aa0','#e69d87','#8dc1a9','#ea7e53','#eedd78','#73a373','#7289ab', '#91ca8c','#f49f42'],
              }
            ]
          });
        }
      }
    }
</script>

<style scoped>
  /*图表*/
  .charts{
    padding-top: 20px;
    display: flex;
    justify-content: space-evenly;
  }
  /*顶层div*/
  .user-mes{
    padding-top: 20px;
    display: flex;
    justify-content: space-between;
  }
  /*用户数的单个div*/
  .user-mes div{
    border-radius: 15px;
    display: flex;
    justify-content: center;
    align-items: center;
  }
  .user-mes span{
    font-size: 18px;
    color: white;
  }
  /*图标大小*/
  .el-icon-s-check,.el-icon-user-solid,.el-icon-s-custom{
    font-size: 40px;
    margin-right: 15px;
    color: white;
  }
</style>
