<template>
    <div>
      <!--上方筛选条件-->
      <div class="select">
        <el-select v-model="value1" multiple placeholder="请选择年级">
          <el-option
            v-for="item in options"
            :key="item.value"
            :label="item.label"
            :value="item.value">
          </el-option>
        </el-select>
        <el-select
          v-model="value2"
          multiple
          collapse-tags
          style="margin-left: 20px;"
          placeholder="请选择专业">
          <el-option
            v-for="item in options1"
            :key="item.value"
            :label="item.label"
            :value="item.value">
          </el-option>
        </el-select>
        <el-select
          v-model="value3"
          multiple
          collapse-tags
          style="margin-left: 20px;"
          placeholder="请选择班级">
          <el-option
            v-for="item in options2"
            :key="item.value"
            :label="item.label"
            :value="item.value">
          </el-option>
        </el-select>
        <el-select
          v-model="value4"
          multiple
          collapse-tags
          style="margin-left: 20px;"
          placeholder="请选择课程">
          <el-option
            v-for="item in options3"
            :key="item.value"
            :label="item.label"
            :value="item.value">
          </el-option>
        </el-select>
        <el-select
          v-model="value5"
          multiple
          collapse-tags
          style="margin-left: 20px;"
          placeholder="请选择分析方法">
          <el-option
            v-for="item in options4"
            :key="item.value"
            :label="item.label"
            :value="item.value">
          </el-option>
        </el-select>
        <el-button type="primary" round @click="query">查询</el-button>
      </div>
      <div class="ranks">
        <!--左边饼状图,分布图-->
        <div class="pie">
          <div id="distribution" :style="{width: '500px', height: '400px'}"></div>
          <div style="padding-bottom: 30px;margin-left: 30px;">
            <el-input placeholder="请输入分数段名称" v-model="input" clearable>
<!--              <template slot="prepend">Http://</template>-->
              <el-button slot="append" icon="el-icon-search" @click="search">搜索</el-button>
            </el-input>
          </div>
        </div>
        <!--右边排行榜-->
        <div class="rank-table">
          <h3>排名</h3>
          <el-table
            :data="tableData"
            border
            style="width: 100%">
            <el-table-column
              prop="name"
              label="姓名"
              width="180">
            </el-table-column>
            <el-table-column
              prop="grade"
              label="成绩"
              width="160">
            </el-table-column>
            <el-table-column
              prop="rank"
              label="排名"
              width="160">
            </el-table-column>
          </el-table>
        </div>

      </div>
    </div>
</template>

<script>
    export default {
        name: "analyze",
      data() {
        return {
          input:'',
          options: [{
            value: '2020级',
            label: '2020级'
          }, {
            value: '2019级',
            label: '2019级'
          }, {
            value: '2018级',
            label: '2018级'
          }, {
            value: '2017级',
            label: '2017级'
          }],
          options1: [{
            value: '物联网',
            label: '物联网'
          }, {
            value: '机械设计制造及自动化',
            label: '机械设计制造及自动化'
          }, {
            value: '计算机科学与技术',
            label: '计算机科学与技术'
          }, {
            value: '电商',
            label: '电商'
          }],
          options2: [{
            value: '1班',
            label: '1班'
          }, {
            value: '2班',
            label: '2班'
          }, {
            value: '3班',
            label: '3班'
          }, {
            value: '4班',
            label: '4班'
          }],
          options3: [{
            value: 'JAVA语言程序设计',
            label: 'JAVA语言程序设计'
          }, {
            value: 'C++语言程序设计',
            label: 'C++语言程序设计'
          }, {
            value: '高等数学A1',
            label: '高等数学A1'
          }, {
            value: '大学物理',
            label: '大学物理'
          }],
          options4: [{
            value: '总分',
            label: '总分'
          }, {
            value: '平均分',
            label: '平均分'
          }],
          value1: [],
          value2: [],
          value3:[],
          value4:[],
          value5:[],
          tableData: [{
            grade: 99,
            name: '王小虎',
            rank: 1
          }, {
            grade: 89,
            name: '李四',
            rank: 2
          }, {
            grade: 85,
            name: '王五',
            rank: 3
          }, {
            grade: 80,
            name: 'xxx',
            rank: 4
          }]
        }
      },
      mounted() {
          this.drawLine();
      },
      methods:{
          //查询按钮
          query(){
            console.log(this.value1,this.value2,this.value3,this.value4,this.value5)
          },
          //搜索按钮，
          search(){
            console.log(this.input);
          },
          //画饼状图
          drawLine(){
            var self = this;
            var detail = this.$echarts.init(document.getElementById('distribution'));
            var option = {
              title: {
                text: '各分数段所占比',
                //调整文字位置
                top:0,
                left:60,
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
              legend: {
                orient: 'vertical',
                right: 'right',
                top:'40px'
              },
              series : [
                {
                  name: '',
                  type: 'pie',    // 设置图表类型为饼图
                  radius: '55%',  // 饼图的半径，外半径为可视区尺寸（容器高宽中较小一项）的 55% 长度。
                  data:[          // 数据数组，name 为数据项名称，value 为数据项值
                    {value:20, name:'0-60'},
                    {value:30, name:'60-80'},
                    {value:40, name:'80-90'},
                    {value:10, name:'90-100'},
                  ],
                  color: ['#73b9bc','#759aa0','#e69d87','#8dc1a9','#ea7e53','#eedd78','#73a373','#7289ab', '#91ca8c','#f49f42'],
                }
              ]
            }
            detail.setOption(option);
          }
      }
    }
</script>

<style scoped>
  .select{
    padding-top: 30px;
    padding-left: 20px;
    display: flex;
    justify-content: space-around;
  }
   .ranks{
     padding-top: 60px;
     display: flex;
     justify-content: space-between;
   }
    .rank-table{
      position: relative;
    }
    .rank-table .el-table{
      margin-top: 35px;
    }
    .rank-table h3{
      font-weight: bold;
      font-size: 18px;
      color: #777;
      position: absolute;
      top: 0;
      left: -140px;
    }
</style>
