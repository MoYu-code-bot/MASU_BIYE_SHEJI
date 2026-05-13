<template>
<!--  就业预测-->
    <div style="padding:20px;height: 480px;">
      <el-transfer
        style="text-align: left; display: inline-block"
        v-model="checked"
        filterable
        :titles="['毕业生就业公司', '预计算列表']"
        :button-texts="['移出', '加入']"
        :format="{
        noChecked: '${total}',
        hasChecked: '${checked}/${total}'
      }"
        @change="handleChange"
        :data="company">
        <span slot-scope="{ option }">{{ option.key }} - {{ option.label }}</span>
        <el-button v-loading="computeLoading" @click="computeData"  type="success" :disabled="checked.length==0" class="transfer-footer" slot="right-footer" size="small">开始计算</el-button>
      </el-transfer>
      <div v-loading="computeLoading" style="float: right;width: 45%;">
        <span>成就-入职率(学生记录数据量少时结果不具备参考价值)</span>
        <el-divider></el-divider>
        <div style="max-height: 420px;overflow-y: auto">
          <h4 v-if="result.length==0" align="center">暂无结果</h4>
          <p v-for="item in result" :key="JSON.stringify(item)" style="margin-bottom: 15px;">
            <el-card shadow="hover" class="box-card">
              <h4 align="right">入职率: <span class="ratio">{{stayTwoNum(item.ratio)}}</span></h4>
              <el-divider></el-divider>
              <span v-for="archive in item.achievements" :key="archive" class="tags">
                <el-tag :type="randomTagType()">{{archive}}</el-tag>
              </span>
            </el-card>
          </p>
        </div>
      </div>
    </div>
</template>

<script>
  import request from '../utils/request'
  import { Notification } from 'element-ui'

  export default {
    name: 'forecast',
    data(){
      return {
        computeLoading:false,
        checked: [1],
        archives:[
          {
            id:1,
            value:'JAVA'
          },
          {
            id:2,
            value:'蓝桥杯'
          },{
            id:3,
            value:'ACM'
          },{
            id:4,
            value:'国家级项目'
          },{
            id:5,
            value:'JAVA'
          }
        ],
        company:[
          {
            key:1,
            label:'国家电网',
            achievements:[1,2]
          },
          {
            key:2,
            label:'蚂蚁金服',
            achievements:[2,3]
          },
          {
            key:3,
            label:'亚信科技',
            achievements:[2,3,4]
          },
          {
            key:4,
            label:'阿里巴巴',
            achievements:[3,5]
          }
        ],
        result:[]
      }
    },
    methods:{
      handleChange(value, direction, movedKeys) {
        console.log(value, direction, movedKeys);
      },
      randomTagType(){
        var colors=['','success','warning','danger'];
        console.log(Math.floor(Math.random()*(colors.length+1)));
        return colors[Math.floor(Math.random()*(colors.length+1))];
      },
      //保留两位小数的百分数
      stayTwoNum(num){
        num=(num*100).toFixed(2);
        if(num==100){
          return "99.99%";
        }
        return num+"%";
      },
      computeData(){
        //得到选中的成就ID
        var data=[];
        for(var i=0;i<this.company.length;i++){
          for(var j=0;j<this.checked.length;j++){
            if(this.company[i].key==this.checked[j]){
              data.push(this.company[i].achievements);
              break;
            }
          }
        }
        //开始请求
        this.computeLoading=true;
        request.post("/api/apriori",{
          minSupport:0.5,
          data:data
        }).then(res=>{
          console.log(res);
          this.computeLoading=false;
          if(res.data.status==0){
            Notification({
              type: 'success',
              message: '计算完成'
            })
            this.createData(res.data.data.data,res.data.data.support);
          }
        });
      },
      createData(data,support){
        this.result=[];
        for(var i=0;i<data.length;i++){
          var names=[];
          for(var j=0;j<data[i].length;j++){
            names.push(this.getArchiveName(data[i][j]));
          }
          this.result.push({
            achievements:names,
            ratio:support[i]
          });
        }
        //排序
        this.result.sort(function(a,b){
          return b.ratio-a.ratio;
        });
      },
      getArchiveName(id){
        for(var i=0;i<this.archives.length;i++){
          if(id==this.archives[i].id){
            return this.archives[i].value;
          }
        }
      }
    }
  }
</script>

<style scoped>
  .tags{
    margin:5px;
    display: inline-block;
  }
  .ratio{
    color:#67C23A;
  }
</style>
