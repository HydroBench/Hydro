//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifndef TIMELIMIT_H
#define TIMELIMIT_H
//


class TimeLimit {
  private:
    double m_orgTime;
    double m_curTime;
    double m_allotedTime;

  protected:
  public:
    // basic constructor
    TimeLimit(void); // default constructor. make it private if needed.
    // TimeLimit();
    // destructor
    ~TimeLimit();
    double timeRemain();
    double timeRemainAll();
    double getTimeAllocated() { return m_allotedTime; };
    // copy operator
    // TimeLimit(const TimeLimit & obj);
    // assignment operator
    // TimeLimit & operator=(const TimeLimit & rhs);
    // access through ()
    // TimeLimit & operator() (int32_t i) ; // lhs
    // TimeLimit & operator() (int32_t i) const ; // rhs
};
#endif
// EOF
