from autobound.autobound.DAG import DAG
from autobound.autobound.Parser import *


def test_collect_worlds():
    dag = DAG()
    dag.from_structure("V -> Z, V -> X, Z -> X, Z -> W, Z -> Y, W -> Y, X -> Y, U -> X, U -> Y", unob = "U")
    test_p = Parser(dag)
    assert test_p.collect_worlds("Y=0") == {'': ['Y=0']}
    assert test_p.collect_worlds("Y(X=1)=0") == {'X=1': ['Y=0']}
    assert test_p.collect_worlds("Y=0&X=0&X(Z=1)=0&V(Z=0,X=0)=0&W(Z=0,X=0)=1&Y(Z=0,X=0)=0") == {'': ['Y=0', 'X=0'], 'Z=1': ['X=0'], 'Z=0,X=0': ['V=0', 'W=1', 'Y=0']}

def test_searchvalue():
    assert (search_value('11010010', '1', (2,2,2)) == np.array([[0,0,0],
                                                               [0,0,1],
                                                               [0,1,1],
                                                               [1,1,0]])).all()
                    

def test_parse_proxy_graph():
    dag = DAG()
    dag.from_structure("W -> X, W -> Y, W -> P, X -> Y", unob = "U")
    parser = Parser(dag)
    assert set(parser.parse('P=0&Y=0&X=0')) == set([('P00', 'W0', 'X00', 'Y0000'), ('P00', 'W0', 'X00', 'Y0001'), ('P00', 'W0', 'X00', 'Y0010'), ('P00', 'W0', 'X00', 'Y0011'), ('P00', 'W0', 'X00', 'Y0100'), ('P00', 'W0', 'X00', 'Y0101'), ('P00', 'W0', 'X00','Y0110'), ('P00', 'W0', 'X00', 'Y0111'), ('P00', 'W0', 'X01', 'Y0000'), ('P00', 'W0', 'X01', 'Y0001'), ('P00', 'W0', 'X01', 'Y0010'), ('P00', 'W0', 'X01', 'Y0011'), ('P00', 'W0', 'X01', 'Y0100'), ('P00', 'W0', 'X01', 'Y0101'), ('P00', 'W0', 'X01', 'Y0110'), ('P00', 'W0', 'X01', 'Y0111'), ('P00', 'W1', 'X00', 'Y0000'), ('P00', 'W1', 'X00', 'Y0001'), ('P00', 'W1', 'X00', 'Y0100'), ('P00', 'W1', 'X00', 'Y0101'), ('P00', 'W1', 'X00', 'Y1000'), ('P00', 'W1', 'X00', 'Y1001'), ('P00', 'W1', 'X00', 'Y1100'), ('P00', 'W1','X00', 'Y1101'), ('P00', 'W1', 'X10', 'Y0000'), ('P00', 'W1', 'X10', 'Y0001'), ('P00', 'W1', 'X10', 'Y0100'), ('P00', 'W1', 'X10', 'Y0101'), ('P00', 'W1', 'X10', 'Y1000'), ('P00', 'W1', 'X10', 'Y1001'), ('P00', 'W1', 'X10', 'Y1100'), ('P00', 'W1', 'X10', 'Y1101'), ('P01', 'W0', 'X00', 'Y0000'), ('P01', 'W0', 'X00', 'Y0001'), ('P01', 'W0', 'X00', 'Y0010'), ('P01', 'W0', 'X00', 'Y0011'), ('P01', 'W0', 'X00', 'Y0100'), ('P01', 'W0', 'X00', 'Y0101'), ('P01', 'W0', 'X00', 'Y0110'), ('P01', 'W0', 'X00', 'Y0111'), ('P01', 'W0', 'X01', 'Y0000'), ('P01', 'W0', 'X01', 'Y0001'), ('P01', 'W0', 'X01', 'Y0010'), ('P01', 'W0', 'X01', 'Y0011'), ('P01', 'W0', 'X01', 'Y0100'), ('P01', 'W0', 'X01', 'Y0101'), ('P01', 'W0', 'X01', 'Y0110'), ('P01', 'W0', 'X01', 'Y0111'), ('P10', 'W1', 'X00', 'Y0000'), ('P10', 'W1', 'X00', 'Y0001'), ('P10', 'W1', 'X00', 'Y0100'), ('P10', 'W1', 'X00', 'Y0101'), ('P10', 'W1', 'X00', 'Y1000'), ('P10', 'W1', 'X00', 'Y1001'), ('P10', 'W1', 'X00', 'Y1100'), ('P10', 'W1', 'X00', 'Y1101'), ('P10', 'W1', 'X10', 'Y0000'),('P10', 'W1', 'X10', 'Y0001'), ('P10', 'W1', 'X10', 'Y0100'), ('P10', 'W1', 'X10', 'Y0101'), ('P10', 'W1', 'X10', 'Y1000'), ('P10', 'W1', 'X10', 'Y1001'), ('P10', 'W1', 'X10', 'Y1100'), ('P10', 'W1', 'X10', 'Y1101')])

def test_parse_iv_graph():
    y = DAG()
    y.from_structure("Z -> X, U -> X, X -> Y, U -> Y", unob = "U , Uy")
    x = Parser(y, {'X': 2})
    assert set(x.parse('Y=1&X=0')) == set([('X00.Y10', 'Z0'), ('X00.Y10', 'Z1'), 
            ('X00.Y11', 'Z0'), ('X00.Y11', 'Z1'), ('X01.Y10', 'Z0'), 
            ('X01.Y11', 'Z0'), ('X10.Y10', 'Z1'), ('X10.Y11', 'Z1')])
    assert x.parse('Y = 1& Y = 0') == [] 
    assert set(x.parse('Y(X=1)=1& Y(X=0)=1')) == set([('X00.Y11',), ('X01.Y11',), ('X10.Y11',), ('X11.Y11',)])
    y = DAG()
    y.from_structure("Z -> Y, U -> X, X -> Y, U -> Y", unob = "U , Uy")
    x = Parser(y, {'X': 2})
    assert set(x.parse('Y(X=1,Z=1)=1')) == set([('X0.Y0001',), ('X0.Y0011',), ('X0.Y0101',), ('X0.Y0111',), ('X0.Y1001',), ('X0.Y1011',), ('X0.Y1101',), ('X0.Y1111',), ('X1.Y0001',), ('X1.Y0011',), ('X1.Y0101',), ('X1.Y0111',), ('X1.Y1001',), ('X1.Y1011',), ('X1.Y1101',), ('X1.Y1111',)])
    assert x.parse('Y=1 &X=1') == [('X1.Y0001', 'Z1'), ('X1.Y0010', 'Z0'), ('X1.Y0011', 'Z0'), 
            ('X1.Y0011', 'Z1'), ('X1.Y0101', 'Z1'), ('X1.Y0110', 'Z0'), ('X1.Y0111', 'Z0'), 
            ('X1.Y0111', 'Z1'), ('X1.Y1001', 'Z1'), ('X1.Y1010', 'Z0'), ('X1.Y1011', 'Z0'),
            ('X1.Y1011', 'Z1'), ('X1.Y1101', 'Z1'), ('X1.Y1110', 'Z0'), ('X1.Y1111', 'Z0'), ('X1.Y1111', 'Z1')] 

def test_parse_irreducible():
    # Testing parse_expr
    y = DAG()
    y.from_structure("Z -> X, U -> X, X -> Y, U -> Y", unob = "U , Uy")
    x = Parser(y, {'X': 2})
    part1 = [('X00.Y11',), ('X01.Y11',), ('X10.Y11',), ('X11.Y11',), ('X00.Y10',), 
            ('X01.Y10',), ('X10.Y10',), ('X11.Y10',)]
    assert set(part1) == set(x.parse_expr('X=0', ['Y=1']))
    part2 = [('X11.Y01',), ('X10.Y01',), ('X01.Y11',), ('X00.Y11',), ('X11.Y11',), 
            ('X10.Y11',), ('X01.Y10',), ('X00.Y10',)] 
    assert set(x.parse_expr('Z=0',['Y=1'])) == set(part2)

